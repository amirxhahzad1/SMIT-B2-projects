"""
LangGraph Agent for StockSense AI.
Orchestrates question validation, tool selection, deterministic tool execution,
result verification, and grounded Gemini answer synthesis.
"""

from typing import TypedDict, Sequence, Optional, Dict, Any, List
import json
import logging
from datetime import date

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

from src.config import config
from src.tools import (
    ALL_TOOLS,
    get_inventory_summary,
    get_expiry_risks,
    get_slow_moving_products,
    get_demand_forecast,
    get_reorder_recommendations,
    get_active_dataset
)

logger = logging.getLogger("StockSenseAI.Agent")


class StockSenseState(TypedDict):
    """Typed state for the StockSense LangGraph workflow."""
    messages: List[BaseMessage]
    question: str
    task_category: Optional[str]
    selected_tool: Optional[str]
    tool_arguments: Optional[Dict[str, Any]]
    tool_result: Optional[Dict[str, Any]]
    clarification_message: Optional[str]
    final_response: str
    error: Optional[str]


def get_gemini_llm() -> Optional[ChatGoogleGenerativeAI]:
    """
    Instantiate and return the configured Gemini chat model.
    Returns None if GOOGLE_API_KEY is not configured.
    """
    if not config.is_gemini_configured():
        return None

    try:
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.1,
            max_retries=2
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
        return None


# -------------------------------------------------------------
# Node 1: Validate User Question
# -------------------------------------------------------------
def validate_question_node(state: StockSenseState) -> Dict[str, Any]:
    """Check if the user question is non-empty and contains a meaningful inquiry."""
    q = state.get("question", "").strip()
    if not q:
        return {
            "clarification_message": "Please enter a specific question about your inventory, expiry risks, slow movers, or demand forecasting.",
            "final_response": "Please enter a question to analyze your inventory."
        }

    if len(q) < 3:
        return {
            "clarification_message": "Your question is very short. Could you provide more detail on what SKU, warehouse, or metric you would like to inspect?",
            "final_response": "Could you clarify what inventory metrics or products you would like me to analyze?"
        }

    return {"clarification_message": None}


# -------------------------------------------------------------
# Node 2: Identify Inventory Task & Select Tool
# -------------------------------------------------------------
SYSTEM_ROUTER_PROMPT = """You are StockSense AI's Intent Router.
Analyze the user's question and determine which inventory tool to execute.

Available Tools:
1. get_inventory_summary: Use for broad portfolio status, overall stock counts, total valuation, or high-level health.
2. get_expiry_risks: Use for questions about expired batches, expiry dates, shelf-life, critical batches, or spoilage risks.
3. get_slow_moving_products: Use for dead stock, slow-moving items, overstocked products, or tied-up capital.
4. get_demand_forecast: Use for projected demand (7d or 30d), stockout risks, runout time, or daily sales velocity.
5. get_reorder_recommendations: Use for replenishment advice, reorder quantities, purchase priority, or supplier lead times.

Respond with a JSON object containing:
{
  "tool_name": "get_inventory_summary" | "get_expiry_risks" | "get_slow_moving_products" | "get_demand_forecast" | "get_reorder_recommendations" | "none",
  "tool_args": { ... any filters like sku, warehouse, risk_category, limit ... },
  "needs_clarification": false,
  "clarification_question": ""
}
"""

def identify_task_node(state: StockSenseState) -> Dict[str, Any]:
    """Classify the inventory question and identify the exact tool and parameters."""
    if state.get("clarification_message"):
        return {}

    question = state["question"]
    llm = get_gemini_llm()

    # Rule-based fallback if LLM is unavailable or for instant deterministic routing
    q_lower = question.lower()
    
    # Try LLM routing first if available
    if llm:
        try:
            routing_prompt = [
                SystemMessage(content=SYSTEM_ROUTER_PROMPT),
                HumanMessage(content=f"User question: {question}")
            ]
            res = llm.invoke(routing_prompt)
            content = res.content
            # Clean markdown codeblocks if returned
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(content)
            tool_name = parsed.get("tool_name")
            tool_args = parsed.get("tool_args", {})
            
            if parsed.get("needs_clarification") and parsed.get("clarification_question"):
                return {
                    "clarification_message": parsed["clarification_question"],
                    "final_response": parsed["clarification_question"]
                }
            
            if tool_name in ["get_inventory_summary", "get_expiry_risks", "get_slow_moving_products", "get_demand_forecast", "get_reorder_recommendations"]:
                return {
                    "selected_tool": tool_name,
                    "tool_arguments": tool_args,
                    "task_category": tool_name.replace("get_", "")
                }
        except Exception as e:
            logger.warning(f"LLM routing failed, falling back to heuristic routing: {e}")

    # Deterministic heuristic classifier
    if any(k in q_lower for k in ["expir", "spoil", "shelf", "critical", "shelf-life", "batch", "lot"]):
        sku_candidate = None
        for word in question.split():
            if "sku" in word.lower():
                sku_candidate = word.strip(".,;:?!'\"")
        return {
            "selected_tool": "get_expiry_risks",
            "tool_arguments": {"sku": sku_candidate} if sku_candidate else {},
            "task_category": "expiry"
        }
    elif any(k in q_lower for k in ["reorder", "replenish", "order", "purchase", "lead time", "buy"]):
        return {
            "selected_tool": "get_reorder_recommendations",
            "tool_arguments": {},
            "task_category": "reorder"
        }
    elif any(k in q_lower for k in ["slow", "dead", "dormant", "overstock", "tied up", "capital", "stagnant"]):
        return {
            "selected_tool": "get_slow_moving_products",
            "tool_arguments": {},
            "task_category": "slow_moving"
        }
    elif any(k in q_lower for k in ["forecast", "demand", "run out", "stockout", "velocity", "sales rate", "days of supply"]):
        return {
            "selected_tool": "get_demand_forecast",
            "tool_arguments": {},
            "task_category": "forecast"
        }
    else:
        return {
            "selected_tool": "get_inventory_summary",
            "tool_arguments": {},
            "task_category": "summary"
        }


# -------------------------------------------------------------
# Node 3: Execute Selected Tool
# -------------------------------------------------------------
def execute_tool_node(state: StockSenseState) -> Dict[str, Any]:
    """Execute the selected deterministic inventory tool with provided arguments."""
    if state.get("clarification_message"):
        return {}

    tool_name = state.get("selected_tool") or "get_inventory_summary"
    tool_args = state.get("tool_arguments") or {}

    tool_map = {
        "get_inventory_summary": get_inventory_summary,
        "get_expiry_risks": get_expiry_risks,
        "get_slow_moving_products": get_slow_moving_products,
        "get_demand_forecast": get_demand_forecast,
        "get_reorder_recommendations": get_reorder_recommendations
    }

    selected_func = tool_map.get(tool_name, get_inventory_summary)

    try:
        # Run tool directly
        result = selected_func.invoke(tool_args)
        return {
            "tool_result": result,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        return {
            "tool_result": None,
            "error": f"Tool execution failed: {str(e)}"
        }


# -------------------------------------------------------------
# Node 4: Verify Tool Results
# -------------------------------------------------------------
def verify_tool_results_node(state: StockSenseState) -> Dict[str, Any]:
    """Ensure tool returned non-null, structured data before passing to LLM."""
    if state.get("clarification_message"):
        return {}

    tool_result = state.get("tool_result")
    if tool_result is None:
        err = state.get("error") or "No data returned from inventory engine."
        return {
            "error": err,
            "final_response": f"I was unable to retrieve data for this query: {err}"
        }

    # Result is verified and valid
    return {"error": None}


# -------------------------------------------------------------
# Node 5: Generate Answer using Gemini
# -------------------------------------------------------------
SYNTHESIS_PROMPT = """You are StockSense AI, an expert inventory intelligence co-pilot for retailers and distributors.
You have been provided with deterministic calculation results from the inventory engine.

RULES:
1. Ground all numerical statements, dates, and calculations exclusively in the verified tool results.
2. Structure your response into:
   - **Direct Answer**: Concise summary answering the user's specific question.
   - **Key Metrics & Evidence**: Clean bullet points or markdown table showing relevant numbers, SKUs, and batches.
   - **Recommended Actions**: 2-3 immediate, actionable operational next steps (e.g. markdowns, reorders, quarantine).
3. If historical data is missing or insufficient for a product, state the limitation honestly. Do not invent numbers.
4. Keep the tone professional, clear, and beginner-friendly.
"""

def generate_answer_node(state: StockSenseState) -> Dict[str, Any]:
    """Synthesize the verified tool output and user question into a grounded response."""
    if state.get("clarification_message"):
        return {"final_response": state["clarification_message"]}

    if state.get("error") and not state.get("tool_result"):
        return {"final_response": f"Analysis halted: {state['error']}"}

    question = state["question"]
    tool_name = state.get("selected_tool", "Unknown Tool")
    tool_result = state.get("tool_result", {})

    llm = get_gemini_llm()
    if not llm:
        # Honest fallback when Gemini API key is missing
        return {
            "final_response": (
                "⚠️ **AI Chat Temporarily Unavailable**: A Google Gemini API key was not detected in `.env` "
                "(or could not be authenticated). \n\n"
                "**Deterministic Tool Result Summary**:\n"
                f"```json\n{json.dumps(tool_result, indent=2)}\n```\n\n"
                "To enable live AI conversational answers, add your `GOOGLE_API_KEY` to the `.env` file and restart."
            )
        }

    try:
        # Compact JSON representation to avoid token bloat
        result_str = json.dumps(tool_result, default=str)

        messages = [
            SystemMessage(content=SYNTHESIS_PROMPT),
            HumanMessage(
                content=(
                    f"User Question: {question}\n\n"
                    f"Executed Tool: {tool_name}\n"
                    f"Verified Tool Output Data:\n{result_str}\n\n"
                    "Generate a concise, well-structured answer with supporting numbers and next actions."
                )
            )
        ]

        response = llm.invoke(messages)
        return {"final_response": response.content}
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        return {
            "error": str(e),
            "final_response": (
                f"⚠️ **Gemini API Error**: Could not generate AI explanation ({str(e)}). \n\n"
                "**Verified Engine Data**:\n"
                f"```json\n{json.dumps(tool_result, indent=2)}\n```"
            )
        }


# -------------------------------------------------------------
# Construct the LangGraph Workflow
# -------------------------------------------------------------
def build_stocksense_graph():
    """Build and compile the StockSense LangGraph state graph."""
    builder = StateGraph(StockSenseState)

    # Add Nodes
    builder.add_node("validate_question", validate_question_node)
    builder.add_node("identify_task", identify_task_node)
    builder.add_node("execute_tool", execute_tool_node)
    builder.add_node("verify_tool_results", verify_tool_results_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Add Edges: Sequential validated pipeline
    builder.add_edge(START, "validate_question")
    builder.add_edge("validate_question", "identify_task")
    builder.add_edge("identify_task", "execute_tool")
    builder.add_edge("execute_tool", "verify_tool_results")
    builder.add_edge("verify_tool_results", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# Singleton compiled graph instance
stocksense_graph = build_stocksense_graph()


def ask_stocksense_agent(question: str, conversation_history: Optional[List[BaseMessage]] = None) -> str:
    """
    Convenience interface to query the StockSense AI agent.
    
    Args:
        question: User inquiry string.
        conversation_history: Optional session conversation history.
        
    Returns:
        Final assistant response string.
    """
    initial_state: StockSenseState = {
        "messages": conversation_history or [HumanMessage(content=question)],
        "question": question,
        "task_category": None,
        "selected_tool": None,
        "tool_arguments": None,
        "tool_result": None,
        "clarification_message": None,
        "final_response": "",
        "error": None
    }

    result = stocksense_graph.invoke(initial_state)
    return result.get("final_response", "No answer generated.")
