import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# Embedding model for RAG indexing. If not set, falls back to the chat model.
# Recommended: an Ollama embedding model like `nomic-embed-text`.
EMBED_MODEL = os.getenv("EMBED_MODEL", OLLAMA_MODEL)

TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

SUMMARY_MAP_CHARS = int(os.getenv("SUMMARY_MAP_CHARS", "7000"))
