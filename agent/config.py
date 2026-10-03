import os
from dotenv import load_dotenv

load_dotenv()

# Pick your model provider: "anthropic" | "openai" | "gemini"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")  # optional override; sensible default picked per-provider below
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "pdf_knowledge_base_openai_1536")
GITHUB_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
GOOGLE_OAUTH_CREDENTIALS = os.getenv("GOOGLE_OAUTH_CREDENTIALS")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "UTC")

# OpenAI embedding model configuration (1536 dimensions)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
MAX_SUPERVISOR_STEPS = 6
