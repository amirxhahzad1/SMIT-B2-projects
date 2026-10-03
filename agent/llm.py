"""LLM Provider Initialization with Automatic Fallbacks & Resilient Error Handling."""
import os
from .config import LLM_PROVIDER, LLM_MODEL

_DEFAULT_MODELS = {
    "anthropic": "claude-3-5-sonnet-20241022",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-flash",
}


def _create_single_llm(provider: str, model_override: str = "", temperature: float = 0):
    """Creates a single LLM instance for a given provider."""
    model = model_override or _DEFAULT_MODELS.get(provider, "gpt-4o-mini")

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not configured in .env")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, max_tokens=4096, api_key=key)

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=key)

    if provider == "gemini":
        key = os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY is not configured in .env")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=key)

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Use 'openai', 'gemini', or 'anthropic'.")


def get_llm(temperature: float = 0):
    """Returns the primary LLM configured in .env with fallback capability if secondary keys exist."""
    primary_llm = _create_single_llm(LLM_PROVIDER, LLM_MODEL, temperature=temperature)

    # Build fallbacks if secondary providers have valid keys configured
    fallbacks = []
    candidates = ["openai", "gemini", "anthropic"]
    for cand in candidates:
        if cand != LLM_PROVIDER:
            try:
                fallbacks.append(_create_single_llm(cand, temperature=temperature))
            except Exception:
                pass  # Skip providers that are not configured

    if fallbacks:
        return primary_llm.with_fallbacks(fallbacks)

    return primary_llm
