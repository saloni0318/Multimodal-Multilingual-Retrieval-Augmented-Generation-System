"""
test_setup.py — Run this to verify your environment is ready.

Run: python test_setup.py

Checks:
  - All Python packages installed
  - Ollama running with required models
  - Qdrant reachable
  - Data directories exist
"""

import sys
from pathlib import Path

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def check(label, fn):
    try:
        result = fn()
        if result is True or result is None:
            print(f"  {PASS} {label}")
            return True
        elif result is False:
            print(f"  {FAIL} {label}")
            return False
        else:
            print(f"  {PASS} {label}: {result}")
            return True
    except Exception as e:
        print(f"  {FAIL} {label}: {e}")
        return False


print("\n=== Multimodal RAG — Setup Check ===\n")

# ── Python packages ───────────────────────────────────────────────────────────
print("Python packages:")
all_ok = True
all_ok &= check("fitz (PyMuPDF)",         lambda: __import__("fitz"))
all_ok &= check("sentence_transformers",  lambda: __import__("sentence_transformers"))
all_ok &= check("qdrant_client",          lambda: __import__("qdrant_client"))
all_ok &= check("ollama",                 lambda: __import__("ollama"))
all_ok &= check("gradio",                 lambda: __import__("gradio"))
all_ok &= check("PIL (Pillow)",           lambda: __import__("PIL"))
all_ok &= check("tqdm",                   lambda: __import__("tqdm"))
all_ok &= check("pandas",                 lambda: __import__("pandas"))

# ── Ollama ────────────────────────────────────────────────────────────────────
print("\nOllama:")
try:
    import ollama as _ollama
    models = _ollama.list()
    names = [m.model for m in models.models]

    has_llava = any("llava" in n for n in names)
    has_gemma = any("gemma3" in n for n in names)

    if has_llava:
        print(f"  {PASS} llava model available")
    else:
        print(f"  {WARN} llava not found — run: ollama pull llava")

    if has_gemma:
        print(f"  {PASS} gemma3 model available")
    else:
        print(f"  {WARN} gemma3 not found — run: ollama pull gemma3")

except Exception as e:
    print(f"  {FAIL} Cannot connect to Ollama: {e}")
    print(f"         Download from: https://ollama.com/download")

# ── Qdrant ────────────────────────────────────────────────────────────────────
print("\nQdrant:")
try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://localhost:6333", timeout=3)
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    print(f"  {PASS} Qdrant running — collections: {names or '(none yet)'}")
except Exception as e:
    print(f"  {WARN} Qdrant not running (needed for Step 3+)")
    print(f"         Start with: docker run -p 6333:6333 qdrant/qdrant")
    print(f"         Or set USE_MEMORY_QDRANT = True in src/config.py")

# ── Data directories ──────────────────────────────────────────────────────────
print("\nData directories:")
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import PDF_DIR, TEXT_DIR, IMAGE_DIR, TABLE_DIR, CAPTION_DIR

for label, d in [
    ("data/pdfs/",                PDF_DIR),
    ("data/extracted/text/",      TEXT_DIR),
    ("data/extracted/images/",    IMAGE_DIR),
    ("data/extracted/tables/",    TABLE_DIR),
    ("data/extracted/captions/",  CAPTION_DIR),
]:
    exists = Path(d).exists()
    print(f"  {PASS if exists else FAIL} {label}")

pdf_count = len(list(PDF_DIR.glob("*.pdf")))
if pdf_count == 0:
    print(f"\n  {WARN} No PDFs found in data/pdfs/ — add PDFs before running the pipeline")
else:
    print(f"\n  {PASS} Found {pdf_count} PDF(s) in data/pdfs/")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*40)
if all_ok:
    print("All Python packages OK!")
    print("\nNext steps:")
    print("  1. Add PDFs to data/pdfs/")
    print("  2. python src/01_extract.py")
    print("  3. python src/02_caption.py")
    print("  4. python src/03_embed_store.py")
    print("  5. python ui/app.py")
else:
    print("Some packages missing. Run:")
    print("  pip install -r requirements.txt")
