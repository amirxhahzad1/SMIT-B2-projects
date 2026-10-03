import asyncio
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import MAX_SUPERVISOR_STEPS
from .guardrails import manage_tool_output
from .llm import get_llm
from .mcp_agents import (CALENDAR_PROMPT, EMAIL_PROMPT, GITHUB_PROMPT,
                         load_calendar_tools, load_email_tools, load_github_tools)
from .rag import RAG_PROMPT, RAG_TOOLS, load_rag_tools
from .utils import dynamic_prompt, message_text


class State(MessagesState):
    steps: int
    instruction: str


class Route(BaseModel):
    """Supervisor decision."""
    next: str = Field(description="Name of the agent to call next, or FINISH")
    instruction: str = Field(default="", description="Precise task for that agent (include all needed details)")
    final_answer: Optional[str] = Field(default=None, description="Reply to the user; required when next is FINISH")


SUPERVISOR_PROMPT = """You are a supervisor coordinating specialist agents:
{roster}

Read the conversation carefully. Pick ONE agent to act next and give it a precise, self-contained instruction
(include dates, names, emails, repo names from the conversation). Agent results come back as messages tagged
'[agent_name result]'. Use several agents in sequence for multi-part requests (e.g. read a calendar, then email
a summary). When the request is fully handled, or you need to ask the user a clarifying question, choose FINISH
and put the complete reply to the user in final_answer. Never invent results; rely on agent outputs.
For plain chit-chat that needs no agent, FINISH directly.

STRICT SHORT & MEANINGFUL OUTPUT RULES:
1. Every final_answer MUST begin with an explicit task status line:
   - If all steps of the user request were successfully performed, start with:
     [Status: COMPLETED]
   - If any step failed, was skipped, or requires user input, start with:
     [Status: NOT COMPLETED - <Reason>]
2. SHORT & MEANINGFUL OUTPUT: Keep final responses strictly short, direct, and meaningful.
   - Limit outputs to 2-4 concise bullet points or a brief 2-3 sentence summary.
   - Remove conversational fluff and filler phrases ("Certainly!", "Sure thing", "I would be glad to help").
   - Focus 100% on core facts, direct results, and actionable outcomes.
3. When a task is COMPLETED, append a brief 1-line note at the end:
   "💡 Note: Ask me anytime if you want to upload/save this output or any file to the RAG knowledge base."
"""


def _make_worker(name: str, agent):
    async def node(state: State):
        msgs = [*state["messages"], HumanMessage(content=f"[Supervisor instruction to {name}]: {state.get('instruction', '')}")]
        try:
            result = await agent.ainvoke({"messages": msgs})
            text = message_text(result["messages"][-1].content)
        except Exception as e:
            text = f"{name} failed: {e}"
        text = manage_tool_output(text)
        return Command(goto="supervisor",
                       update={"messages": [HumanMessage(content=f"[{name} result]\n{text}", name=name)]})
    return node


async def _try_load(label, loader):
    try:
        tools = await loader()
        print(f"  ✓ {label}: {len(tools)} tools")
        return tools
    except Exception as e:
        print(f"  ✗ {label} disabled: {e}")
        return None


async def build_graph():
    llm = get_llm()
    print("Loading agents...")

    agents = {}
    for name, desc, loader, prompt in [
        ("rag_agent", "Answers questions about ingested PDFs & docs; can upload local files (PDF/TXT/MD) & save completed task results into RAG vector store.", load_rag_tools, RAG_PROMPT),
        ("github_agent", "Any GitHub question: repos, code, issues, PRs, commits, workflows.", load_github_tools, GITHUB_PROMPT),
        ("calendar_agent", "Google Calendar: read meetings, check availability, create/update/delete events.", load_calendar_tools, CALENDAR_PROMPT),
        ("email_agent", "Gmail: write emails, save drafts, send now, schedule sends, search inbox.", load_email_tools, EMAIL_PROMPT),
    ]:
        tools = await _try_load(name, loader)
        if tools:
            agents[name] = (desc, create_react_agent(llm, tools, prompt=dynamic_prompt(prompt)))

    roster = "\n".join(f"- {n}: {d}" for n, (d, _) in agents.items())
    router = llm.with_structured_output(Route)
    valid = set(agents)

    async def supervisor(state: State):
        steps = state.get("steps", 0)
        try:
            decision: Route = await router.ainvoke(
                [SystemMessage(content=SUPERVISOR_PROMPT.format(roster=roster)), *state["messages"]]
            )
            if decision.next in valid and steps < MAX_SUPERVISOR_STEPS:
                return Command(goto=decision.next, update={"steps": steps + 1, "instruction": decision.instruction})

            answer = decision.final_answer or ""
            if not answer.startswith("[Status:"):
                if steps >= MAX_SUPERVISOR_STEPS:
                    status_prefix = "[Status: NOT COMPLETED - Maximum execution steps reached]\n\n"
                else:
                    status_prefix = "[Status: COMPLETED]\n\n"
                answer = status_prefix + (answer or "Request processed.")

            return Command(goto=END, update={"messages": [AIMessage(content=answer)]})
        except Exception as e:
            err_answer = f"[Status: NOT COMPLETED - System error]\n\nSupervisor encountered an issue: {e}"
            return Command(goto=END, update={"messages": [AIMessage(content=err_answer)]})

    g = StateGraph(State)
    g.add_node("supervisor", supervisor)
    for n, (_, agent) in agents.items():
        g.add_node(n, _make_worker(n, agent))
    g.add_edge(START, "supervisor")
    return g.compile(checkpointer=MemorySaver())
