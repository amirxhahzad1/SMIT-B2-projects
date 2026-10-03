import asyncio
import sys
import warnings

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_orig_showwarning = warnings.showwarning
def _custom_showwarning(message, category, filename, lineno, file=None, line=None):
    if "allowed_objects" in str(message):
        return
    _orig_showwarning(message, category, filename, lineno, file, line)
warnings.showwarning = _custom_showwarning

from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.guardrails import apply_output_guardrails, validate_input_guardrails
from agent.scheduler_tool import ensure_scheduler_started
from agent.utils import message_text


async def main():
    print("=" * 60)
    print(" 🚀 LangGraph Multi-Agent Assistant CLI")
    print(" Services: GitHub, Google Calendar, Gmail & PDF Knowledge Base")
    print("=" * 60)

    try:
        graph = await build_graph()
    except Exception as e:
        print(f"\n[System Error] Failed to initialize agent graph: {e}\n")
        return

    ensure_scheduler_started()  # background scheduler for "send email later"
    config = {"configurable": {"thread_id": "cli-session"}, "recursion_limit": 60}

    print("\nAssistant is ready! Type your request below (e.g. 'what are your services', 'send email to...', 'check my calendar').")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            q = (await asyncio.to_thread(input, "you> ")).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        # 1. Input Guardrails Check
        is_valid, validation_msg = validate_input_guardrails(q)
        if not is_valid:
            print(f"\nassistant> [Guardrail Warning] {validation_msg}\n")
            continue

        final = None
        try:
            async for update in graph.astream({"messages": [HumanMessage(content=q)], "steps": 0},
                                              config, stream_mode="updates"):
                for node, out in update.items():
                    if node not in ("supervisor",) and out:
                        print(f"  → {node} done")
                    if node == "supervisor" and out and out.get("messages"):
                        final = message_text(out["messages"][-1].content)
        except Exception as err:
            final = f"[Error Service] An error occurred while processing your request: {err}"

        # 2. Output Guardrails Check & Sanitize
        sanitized_final = apply_output_guardrails(final or "Completed processing.")
        print(f"\nassistant> {sanitized_final}\n")


if __name__ == "__main__":
    asyncio.run(main())
