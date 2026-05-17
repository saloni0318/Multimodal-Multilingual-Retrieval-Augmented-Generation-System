"""
04_query.py — Step 4: Query the RAG system.

Run: python src/04_query.py
Or import RAGEngine into the Gradio UI.

What it does:
- Embeds the user's query with multilingual-e5-large
- Retrieves top-k chunks from Qdrant (text + table + image captions)
- Passes retrieved context to Gemma3 via Ollama
- Gemma3 answers in the same language as the query
- Returns the answer + source citations (with image paths where relevant)
"""

import sys
from langdetect import detect
from pathlib import Path
from typing import Optional

import ollama
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EMBED_MODEL_NAME, QDRANT_URL, QDRANT_COLLECTION,
    USE_MEMORY_QDRANT, TOP_K, SCORE_THRESHOLD, LLM_MODEL,
)

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ── System prompt for the LLM ────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a helpful multilingual assistant that answers questions
based on documents. You will be given relevant excerpts from documents including
text, table data, and descriptions of images/charts.

Rules:

1. Answer ONLY based on the provided context.
   If the answer isn't in the context, say so.

2. Always answer in the SAME language as the user's question.
   If im writing in english answer me in english,if malayalam then give answer in malayalam   same for other languages as well support all languages of India.
   Never switch languages.

3. When referencing a chart or image, mention it explicitly
   (e.g., "As shown in the chart on page X...").

4. When referencing table data, present it clearly.

5. Cite the source and page number at the end of your answer.

6. Be concise but complete.
"""

# ── RAGEngine class (reusable by UI and CLI) ─────────────────────────────────

class RAGEngine:
    def __init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._qdrant: Optional[QdrantClient] = None

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"Loading embedding model: {EMBED_MODEL_NAME}...")
            self._embedder = SentenceTransformer(EMBED_MODEL_NAME)
            print("  Ready.")
        return self._embedder

    def _get_qdrant(self) -> QdrantClient:
        if self._qdrant is None:
            if USE_MEMORY_QDRANT:
                self._qdrant = QdrantClient(":memory:")
            else:
                self._qdrant = QdrantClient(url=QDRANT_URL)
        return self._qdrant

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using the multilingual-e5 prefix convention."""
        model = self._get_embedder()
        # Use "query: " prefix for search queries (as per e5 training)
        vector = model.encode(
            f"query: {query}",
            normalize_embeddings=True
        )
        return vector.tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        modality_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Embed the query and retrieve top-k chunks from Qdrant.

        Args:
            query: user's question in any language
            top_k: number of results to return
            modality_filter: optional — "text", "table", or "image"

        Returns:
            List of dicts with keys: text, source, page, modality, image_path, score
        """
        vector = self.embed_query(query)
        client = self._get_qdrant()

        # Build optional modality filter
        query_filter = None
        if modality_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="modality",
                        match=MatchValue(value=modality_filter),
                    )
                ]
            )

        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=SCORE_THRESHOLD,
            with_payload=True,
        ).points

        chunks = []
        for r in results:
            chunks.append({
                "text":       r.payload.get("text", ""),
                "source":     r.payload.get("source", ""),
                "page":       r.payload.get("page", 0),
                "modality":   r.payload.get("modality", "text"),
                "image_path": r.payload.get("image_path", ""),
                "score":      round(r.score, 4),
            })

        return chunks

    def build_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            modality = chunk["modality"]
            source   = chunk["source"]
            page     = chunk["page"]
            text     = chunk["text"]

            if modality == "image":
                header = f"[Source {i} | IMAGE/CHART | {source} | Page {page}]"
                parts.append(f"{header}\nImage description: {text}")
            elif modality == "table":
                header = f"[Source {i} | TABLE | {source} | Page {page}]"
                parts.append(f"{header}\nTable row: {text}")
            else:
                header = f"[Source {i} | TEXT | {source} | Page {page}]"
                parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    def generate_answer(self, query: str, context: str) -> str:
        """Send context + query to Gemma3 via Ollama."""

        # Dynamically detect user language
        try:
            user_lang = detect(query)
        except Exception:
            user_lang = "en"

        prompt = f"""
    IMPORTANT INSTRUCTION:

    The detected language of the user is: {user_lang}

    You MUST:
    - Answer ONLY in {user_lang}
    - Never translate
    - Never switch languages
    - Never answer in Portuguese unless the query is Portuguese
    - Preserve the user's writing style

    Context from documents:
    {context}

    User question:
    {query}

    Answer:
    """

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )

            return response["message"]["content"].strip()

        except Exception as e:
            return (
                f"Error generating answer: {e}\n\n"
                f"Make sure Ollama is running and "
                f"'{LLM_MODEL}' is pulled.\n"
                f"Run: ollama pull {LLM_MODEL}"
            )
    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        modality_filter: Optional[str] = None,
    ) -> dict:
        """
        Full RAG pipeline: retrieve + generate.

        Returns:
            {
                "answer":  str,
                "sources": list[dict],   # retrieved chunks with scores
            }
        """
        # 1. Retrieve relevant chunks
        chunks = self.retrieve(question, top_k=top_k, modality_filter=modality_filter)

        if not chunks:
            return {
                "answer": "No relevant content found in the documents for your query.",
                "sources": [],
            }

        # 2. Build context string
        context = self.build_context(chunks)

        # 3. Generate answer
        answer = self.generate_answer(question, context)

        return {
            "answer":  answer,
            "sources": chunks,
        }


# ── CLI for quick testing ─────────────────────────────────────────────────────

def check_ollama():
    try:
        models = ollama.list()
        names = [m["name"] for m in models.get("models", [])]
        has_llm = any(LLM_MODEL in n for n in names)
        if not has_llm:
            print(f"WARNING: '{LLM_MODEL}' not found in Ollama.")
            print(f"Run: ollama pull {LLM_MODEL}")
            return False
        return True
    except Exception:
        print("WARNING: Ollama not running. Answer generation will fail.")
        print("Start Ollama from: https://ollama.com/download")
        return False


def main():
    print("=== Multimodal Multilingual RAG — Query Mode ===\n")
    check_ollama()

    engine = RAGEngine()

    print("Type your question in any language. Type 'quit' to exit.")
    print("Tip: prefix with [image], [table], or [text] to filter by modality.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        # Check for modality filter prefix
        modality_filter = None
        for mod in ["image", "table", "text"]:
            if question.lower().startswith(f"[{mod}]"):
                modality_filter = mod
                question = question[len(f"[{mod}]"):].strip()
                print(f"  Filtering by modality: {mod}")
                break

        print("\nSearching...", flush=True)
        result = engine.query(question, modality_filter=modality_filter)

        print(f"\n{'─'*60}")
        print("ANSWER:")
        print(result["answer"])

        print(f"\nSOURCES ({len(result['sources'])} retrieved):")
        for i, src in enumerate(result["sources"], 1):
            icon = {"text": "T", "table": "TB", "image": "IMG"}.get(src["modality"], "?")
            print(f"  [{icon}] {src['source']} | page {src['page']} | score: {src['score']}")
            if src["modality"] == "image" and src["image_path"]:
                print(f"       Image: {src['image_path']}")

        print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
