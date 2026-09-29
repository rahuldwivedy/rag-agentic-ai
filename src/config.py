import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 3072  # default output size of gemini-embedding-001
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
MIN_RELEVANCE = 0.50  # top similarity below this => out-of-scope; tune after testing
REFUSAL = "I cannot answer based on the provided document."
BATCH_SIZE = 20       # chunks per upsert batch (free-tier rate limits)
BATCH_PAUSE = 20       # seconds between batches
