"""
config.py — All settings in one place.
Change things here, everything else picks them up automatically.
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data"
PDF_DIR       = DATA_DIR / "pdfs"
EXTRACT_DIR   = DATA_DIR / "extracted"
TEXT_DIR      = EXTRACT_DIR / "text"
IMAGE_DIR     = EXTRACT_DIR / "images"
TABLE_DIR     = EXTRACT_DIR / "tables"
CAPTION_DIR   = EXTRACT_DIR / "captions"

# Create all dirs if they don't exist
for d in [PDF_DIR, TEXT_DIR, IMAGE_DIR, TABLE_DIR, CAPTION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding model (free, local, 100+ languages) ────────────────────────────
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_DIM        = 1024   # multilingual-e5-large output dimension

# ── Ollama models (free, local) ──────────────────────────────────────────────
VISION_MODEL  = "llava"       # For image captioning
LLM_MODEL     = "gemma3"      # For answer generation (good Indic lang support)
OLLAMA_URL    = "http://localhost:11434"

# ── Qdrant vector store ──────────────────────────────────────────────────────
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "multimodal_rag_docs"

# Set to True to use in-memory Qdrant (no Docker needed, data lost on restart)
USE_MEMORY_QDRANT = True

# ── Chunking settings ────────────────────────────────────────────────────────
CHUNK_SIZE    = 512    # characters per text chunk
CHUNK_OVERLAP = 64     # overlap between chunks

# ── Retrieval settings ───────────────────────────────────────────────────────
TOP_K           = 6    # number of chunks to retrieve
SCORE_THRESHOLD = 0.3  # minimum similarity score to include

# ── Image extraction settings ────────────────────────────────────────────────
MIN_IMAGE_WIDTH  = 100  # pixels — skip tiny images (icons, bullets)
MIN_IMAGE_HEIGHT = 100
IMAGE_DPI        = 150  # DPI when rasterising pages for image extraction
