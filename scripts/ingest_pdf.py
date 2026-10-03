"""Usage: python -m scripts.ingest_pdf path/to/file.pdf [more.pdf ...]"""
import sys

from agent.rag import ingest_pdf

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"{p}: {ingest_pdf(p)} chunks ingested")
