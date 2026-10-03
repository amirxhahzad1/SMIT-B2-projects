"""RAG sub-agent: PDFs & Documents -> chunks -> Gemini Embeddings -> hosted Qdrant Cloud -> retrieval & upload tools."""
import hashlib
import os
import uuid

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (EMBEDDING_DIM, EMBEDDING_MODEL, EMBEDDING_PROVIDER,
                     QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL)

_client = None
_store = None


def get_embeddings_model():
    """Initializes embedding model based on EMBEDDING_PROVIDER ('openai' or 'gemini')."""
    if EMBEDDING_PROVIDER == "openai":
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is not configured in .env for OpenAI embeddings.")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_key)

    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured in .env for Gemini embeddings.")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_key)


def _get_client():
    global _client
    if _client is None:
        if not QDRANT_URL:
            raise RuntimeError("QDRANT_URL is not set in .env")
        from qdrant_client import QdrantClient
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


def get_vector_store():
    global _store
    if _store is None:
        from qdrant_client.http import models as qm
        from langchain_qdrant import QdrantVectorStore

        client = _get_client()
        embeddings = get_embeddings_model()
        if client.collection_exists(QDRANT_COLLECTION):
            info = client.get_collection(QDRANT_COLLECTION)
            existing_size = getattr(info.config.params.vectors, 'size', None)
            if existing_size and existing_size != EMBEDDING_DIM:
                client.recreate_collection(
                    QDRANT_COLLECTION,
                    vectors_config=qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
                )
        else:
            client.create_collection(
                QDRANT_COLLECTION,
                vectors_config=qm.VectorParams(size=EMBEDDING_DIM, distance=qm.Distance.COSINE),
            )
            client.create_payload_index(
                QDRANT_COLLECTION, "metadata.source", qm.PayloadSchemaType.KEYWORD
            )
        _store = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )
    return _store


def ingest_file(path: str) -> int:
    """Load a local PDF, text, or markdown file, split it, embed it with Gemini embeddings, and upsert into Qdrant."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found at: {path}")

    name = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        pages = loader.load()
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(path, encoding="utf-8", autodetect_encoding=True)
        pages = loader.load()

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(pages)
    ids = []
    for c in chunks:
        c.metadata["source"] = name
        digest = hashlib.sha1(f"{name}|{c.metadata.get('page', 0)}|{c.page_content}".encode()).hexdigest()
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, digest)))

    get_vector_store().add_documents(chunks, ids=ids)
    return len(chunks)


@tool
def search_pdf_knowledge_base(query: str, source: str | None = None, k: int = 5) -> str:
    """Semantic search over ingested documents using Gemini Embeddings. Optionally restrict to one document via `source`
    (the file name, e.g. 'report.pdf'). Returns passages with file name and page number."""
    from qdrant_client.http import models as qm
    flt = None
    if source:
        flt = qm.Filter(must=[qm.FieldCondition(key="metadata.source", match=qm.MatchValue(value=source))])
    docs = get_vector_store().similarity_search(query, k=k, filter=flt)
    if not docs:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{d.metadata.get('source')} | page {int(d.metadata.get('page', 0)) + 1}]\n{d.page_content}"
        for d in docs
    )


@tool
def list_ingested_documents() -> str:
    """List all file names and task outputs currently stored in the vector database."""
    client, seen, offset = _get_client(), set(), None
    while True:
        points, offset = client.scroll(
            QDRANT_COLLECTION, limit=256, offset=offset, with_payload=["metadata.source"], with_vectors=False
        )
        for p in points:
            src = (p.payload or {}).get("metadata", {}).get("source")
            if src:
                seen.add(src)
        if offset is None:
            break
    return "\n".join(sorted(seen)) or "The knowledge base is empty."


@tool
def upload_document_file(file_path: str) -> str:
    """Upload/ingest a local file (PDF, TXT, MD) into the RAG vector database so it can be searched later."""
    try:
        count = ingest_file(file_path)
        return f"Successfully uploaded and ingested {count} chunks from '{os.path.basename(file_path)}' into the RAG knowledge base."
    except Exception as e:
        return f"File upload/ingestion failed: {e}"


@tool
def upload_task_result_to_rag(task_title: str, content: str) -> str:
    """Upload and store a completed task result, report, summary, or note directly into the RAG knowledge base."""
    try:
        doc_name = f"task_{task_title.lower().replace(' ', '_')}.txt"
        doc = Document(page_content=content, metadata={"source": doc_name, "title": task_title})
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents([doc])
        ids = []
        for i, c in enumerate(chunks):
            digest = hashlib.sha1(f"{doc_name}|{i}|{c.page_content}".encode()).hexdigest()
            ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, digest)))
        get_vector_store().add_documents(chunks, ids=ids)
        return f"Task result '{task_title}' uploaded successfully as '{doc_name}' ({len(chunks)} chunks ingested)."
    except Exception as e:
        return f"Failed to upload task result to RAG: {e}"


# Keep backward compatibility
ingest_pdf_file = upload_document_file

RAG_TOOLS = [
    search_pdf_knowledge_base,
    list_ingested_documents,
    upload_document_file,
    upload_task_result_to_rag,
]


async def load_rag_tools():
    """Validates vector store and Gemini embeddings initialization before returning tools."""
    get_vector_store()
    return RAG_TOOLS


RAG_PROMPT = """You are the Document & Knowledge-Base specialist powered by Gemini Embeddings.
- You can search document content, list ingested files, upload local files (PDF/TXT/MD), and upload completed task results directly into the vector database.
- Always call search_pdf_knowledge_base before answering questions about document content.
- Answer ONLY from retrieved passages; cite as (file, page). If the passages don't contain the answer, say so.
- Use list_ingested_documents when asked what files are available.
- Use upload_document_file when the user wants to upload a file into the knowledge base.
- Use upload_task_result_to_rag when asked to save/upload a completed task output, report, or summary into the knowledge base."""
