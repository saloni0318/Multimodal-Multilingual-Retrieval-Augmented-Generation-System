"""
03_embed_store.py — Step 3: Embed all chunks and store in Qdrant.
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TEXT_DIR, TABLE_DIR, CAPTION_DIR,
    EMBED_MODEL_NAME, EMBED_DIM,
    QDRANT_URL, QDRANT_COLLECTION, USE_MEMORY_QDRANT,
)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct,
)

# ── Global shared client and model (reused across calls in same process) ──────
_shared_client: Optional[QdrantClient] = None
_shared_model: Optional[SentenceTransformer] = None


def load_embedder():
    global _shared_model
    if _shared_model is not None:
        return _shared_model
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    _shared_model = SentenceTransformer(EMBED_MODEL_NAME)
    print("  Model loaded.")
    return _shared_model


def get_qdrant_client() -> QdrantClient:
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    if USE_MEMORY_QDRANT:
        print("Using in-memory Qdrant")
        _shared_client = QdrantClient(":memory:")
    else:
        print(f"Connecting to Qdrant at {QDRANT_URL}")
        _shared_client = QdrantClient(url=QDRANT_URL)
        _shared_client.get_collections()
        print("  Connected.")
    return _shared_client


def setup_collection(client: QdrantClient, recreate: bool = True):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        if recreate:
            print(f"Recreating collection '{QDRANT_COLLECTION}'...")
            client.delete_collection(QDRANT_COLLECTION)
        else:
            print(f"Collection already exists.")
            return
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection '{QDRANT_COLLECTION}'")


def embed_batch(model, texts):
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def load_all_chunks():
    all_chunks = []

    for json_file in TEXT_DIR.glob("*_text.json"):
        chunks = json.loads(json_file.read_text(encoding='utf-8'))
        all_chunks.extend(chunks)
    print(f"  Text chunks     : {sum(1 for c in all_chunks if c.get('modality') == 'text')}")

    for json_file in TABLE_DIR.glob("*_tables.json"):
        rows = json.loads(json_file.read_text(encoding='utf-8'))
        all_chunks.extend(rows)
    print(f"  Table rows      : {sum(1 for c in all_chunks if c.get('modality') == 'table')}")

    caption_count = 0
    for json_file in CAPTION_DIR.glob("*_captions.json"):
        items = json.loads(json_file.read_text(encoding='utf-8'))
        for item in items:
            caption = item.get("caption", "")
            if caption and not caption.startswith("["):
                all_chunks.append({
                    "text":       caption,
                    "source":     item["source"],
                    "page":       item["page"],
                    "modality":   "image",
                    "image_path": item.get("image_path", ""),
                    "img_idx":    item.get("img_idx", 0),
                })
                caption_count += 1
    print(f"  Image captions  : {caption_count}")
    return all_chunks


def store_chunks(client, model, chunks):
    BATCH_SIZE = 64
    total = len(chunks)
    print(f"\nEmbedding and storing {total} chunks...")

    for batch_start in tqdm(range(0, total, BATCH_SIZE)):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        vectors = embed_batch(model, texts)

        points = []
        for chunk, vector in zip(batch, vectors):
            payload = {
                "text":       chunk["text"],
                "source":     chunk.get("source", ""),
                "page":       chunk.get("page", 0),
                "modality":   chunk.get("modality", "text"),
                "image_path": chunk.get("image_path", ""),
            }
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            ))
        client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    print(f"  Stored {total} chunks in Qdrant.")


def embed_and_store() -> QdrantClient:
    """
    KEY FUNCTION — called directly from ui/app.py in the SAME process.
    Returns the shared QdrantClient so RAGEngine can reuse it.
    """
    global _shared_client

    print("\nLoading all chunks...")
    chunks = load_all_chunks()

    if not chunks:
        print("No chunks to embed.")
        return None

    model = load_embedder()
    client = get_qdrant_client()
    setup_collection(client, recreate=True)
    store_chunks(client, model, chunks)

    count = client.get_collection(QDRANT_COLLECTION).points_count
    print(f"Verification: {count} vectors stored.")
    return client


def main():
    print("=== Step 3: Embed + Store ===\n")
    text_files    = list(TEXT_DIR.glob("*_text.json"))
    table_files   = list(TABLE_DIR.glob("*_tables.json"))
    caption_files = list(CAPTION_DIR.glob("*_captions.json"))

    if not text_files and not table_files and not caption_files:
        print("No extracted data found. Run python src/01_extract.py first.")
        return

    client = embed_and_store()
    if client:
        print(f"\nNext step: python src/04_query.py")


if __name__ == "__main__":
    main()