"""Input & Output Guardrails, Credential Redaction, and Lengthy Output Management module."""
import re
from typing import Tuple
from langchain_core.messages import AIMessage, HumanMessage


# Common prompt injection & jailbreak patterns to intercept at input boundary
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|above)\s+instructions",
    r"bypass\s+(safety|security|guardrails|filters)",
    r"reveal\s+(system\s+prompt|instructions|secret|api_key|credentials)",
    r"you\s+are\s+now\s+in\s+DAN\s+mode",
    r"override\s+system\s+rules",
]

# Sensitive patterns to redact from agent outputs (API Keys, Tokens, Credentials)
SENSITIVE_CREDENTIAL_PATTERNS = [
    (r"sk-proj-[A-Za-z0-9_-]{30,}", "[REDACTED_OPENAI_KEY]"),
    (r"sk-ant-[A-Za-z0-9_-]{30,}", "[REDACTED_ANTHROPIC_KEY]"),
    (r"ghp_[A-Za-z0-9]{36}", "[REDACTED_GITHUB_PAT]"),
    (r"github_pat_[A-Za-z0-9_]{40,}", "[REDACTED_GITHUB_PAT]"),
    (r"AQ\.[A-Za-z0-9_-]{40,}", "[REDACTED_GOOGLE_KEY]"),
    (r"eyJhbGciOi[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", "[REDACTED_JWT_TOKEN]"),
]


def validate_input_guardrails(text: str, max_length: int = 4000) -> Tuple[bool, str]:
    """Input Guardrail: Validates user prompt for length, non-emptiness, and prompt injection."""
    if not text or not text.strip():
        return False, "Input cannot be empty."

    if len(text) > max_length:
        return False, f"Input exceeds maximum allowed length of {max_length} characters."

    # Check for prompt injection attempts
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "Request blocked by Security Guardrails: Prompt injection or system prompt override attempt detected."

    return True, text.strip()


def manage_lengthy_output(text: str, max_chars: int = 3000, max_lines: int = 60) -> str:
    """Manages lengthy outputs by capping max length/lines with clean truncation markers."""
    if not text:
        return text

    lines = text.split("\n")
    needs_char_truncation = len(text) > max_chars
    needs_line_truncation = len(lines) > max_lines

    if not (needs_char_truncation or needs_line_truncation):
        return text

    # Handle line count truncation first if needed
    if needs_line_truncation:
        top_lines = lines[:35]
        bottom_lines = lines[-15:]
        truncated_line_count = len(lines) - 50
        text = "\n".join(top_lines) + f"\n\n--- [Output Managed: {truncated_line_count} middle lines truncated for readability] ---\n\n" + "\n".join(bottom_lines)

    # Handle character count truncation
    if len(text) > max_chars:
        head_length = int(max_chars * 0.65)
        tail_length = int(max_chars * 0.25)
        omitted_chars = len(text) - (head_length + tail_length)
        head = text[:head_length]
        tail = text[-tail_length:]
        text = f"{head}\n\n--- [Output Managed: {omitted_chars:,} characters truncated for readability] ---\n[Tip: Ask for specific details or save the complete output to a file or RAG knowledge base.]\n\n{tail}"

    return text


def manage_tool_output(text: str, max_chars: int = 2500) -> str:
    """Caps internal worker tool outputs to prevent bloated graph state context."""
    if not text or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n... [Tool output truncated: {omitted:,} characters omitted]"


def apply_output_guardrails(text: str, max_chars: int = 3000) -> str:
    """Output Guardrail: Redacts API keys, credentials, and manages lengthy output display."""
    if not text:
        return "I couldn't process your request at this time. Please try again."

    sanitized = text
    # Redact sensitive credentials
    for pattern, replacement in SENSITIVE_CREDENTIAL_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized)

    # Manage lengthy output
    managed = manage_lengthy_output(sanitized, max_chars=max_chars)
    return managed
