from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import SystemMessage

from .config import DEFAULT_TIMEZONE


def message_text(content) -> str:
    """Normalise LangChain message content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def dynamic_prompt(base_prompt: str):
    """Return a prompt callable that injects the *current* time on every model call."""

    def _prompt(state):
        now = datetime.now(ZoneInfo(DEFAULT_TIMEZONE))
        header = (
            f"{base_prompt}\n\nCurrent date/time: {now.isoformat(timespec='minutes')} "
            f"({DEFAULT_TIMEZONE}). Resolve relative dates like 'tomorrow' or "
            f"'next Monday' against this."
        )
        return [SystemMessage(content=header), *state["messages"]]

    return _prompt
