"""
ui/app.py — Gradio web interface with PDF upload support.
Run: python ui/app.py
"""

import sys
import shutil
import subprocess
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import LLM_MODEL, QDRANT_COLLECTION, PDF_DIR

# Import embed module directly
import importlib.util

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

SRC = Path(__file__).parent.parent / "src"
embed_mod = _load_module("embed", SRC / "03_embed_store.py")
query_mod = _load_module("query", SRC / "04_query.py")

RAGEngine = query_mod.RAGEngine

# ── Shared state ──────────────────────────────────────────────────────────────
engine = RAGEngine()
_qdrant_client = None  # will be set after upload


def _reinit_engine(client):
    """Point the engine's Qdrant to the freshly created in-memory client."""
    global _qdrant_client
    _qdrant_client = client
    engine._qdrant = client


# ── PDF Upload + Pipeline ─────────────────────────────────────────────────────

def process_uploaded_pdf(pdf_file):
    if pdf_file is None:
        yield "No file uploaded."
        return

    try:
        # 1. Clear old data
        extracted_base = PDF_DIR.parent / "extracted"
        for folder in ["text", "images", "tables", "captions"]:
            for f in (extracted_base / folder).glob("*"):
                if f.is_file():
                    f.unlink()
        for f in PDF_DIR.glob("*.pdf"):
            f.unlink()
        yield "Old data cleared!\nCopying new PDF..."

        # 2. Copy new PDF
        src_path = Path(pdf_file)
        dest_path = PDF_DIR / src_path.name
        shutil.copy(src_path, dest_path)
        yield f"PDF copied: {src_path.name}\nExtracting text and images..."

        # 3. Extract (subprocess is fine — just writes files to disk)
        result = subprocess.run(
            [sys.executable, str(SRC / "01_extract.py")],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            yield f"Extraction failed:\n{result.stderr}"
            return
        yield "Extraction done!\nCaptioning images (few minutes)..."

        # 4. Caption (subprocess is fine — just writes files to disk)
        result = subprocess.run(
            [sys.executable, str(SRC / "02_caption.py")],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            yield f"Captioning failed:\n{result.stderr}"
            return
        yield "Captioning done!\nEmbedding into vector database..."

        # 5. Embed IN THE SAME PROCESS — this is the key fix!
        client = embed_mod.embed_and_store()
        if client is None:
            yield "Embedding failed — no chunks found."
            return

        # 6. Give the engine the same in-memory client
        _reinit_engine(client)

        yield f"All done! '{src_path.name}' is ready.\nAsk your questions now!"

    except Exception as e:
        import traceback
        yield f"Error: {e}\n{traceback.format_exc()}"


# ── Chat helpers ──────────────────────────────────────────────────────────────

def format_sources(sources):
    if not sources:
        return "_No sources retrieved._"
    lines = []
    for i, src in enumerate(sources, 1):
        mod  = src["modality"]
        icon = {"text": "T", "table": "TB", "image": "IMG"}.get(mod, "?")
        lines.append(
            f"[{icon}] Source {i} | {src['source']} | "
            f"Page {src['page']} | Score: {src['score']}"
        )
    return "\n\n".join(lines)


def collect_images(sources):
    images = []
    for src in sources:
        if src["modality"] == "image" and src.get("image_path"):
            img_path = Path(src["image_path"])
            if img_path.exists():
                try:
                    images.append((Image.open(img_path), f"{src['source']} | Page {src['page']}"))
                except Exception:
                    pass
    return images


def chat(message, history, modality_filter, top_k):
    if not message.strip():
        return history, "", [], "_No sources._"

    if _qdrant_client is None:
        history = history or []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "Please upload a PDF first using Step 1 above."})
        return history, "", [], "_No sources._"

    mod_filter = None if modality_filter == "All" else modality_filter.lower()
    result = engine.query(message, top_k=top_k, modality_filter=mod_filter)

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": result["answer"]})
    return history, "", collect_images(result["sources"]), format_sources(result["sources"])

# ── Example queries ───────────────────────────────────────────────────────────

EXAMPLES = [
    ["What are the key findings in this document?"],
    ["Explain the main algorithm used"],
    ["What datasets were used in experiments?"],
    ["4-ാം പേജിലെ ചാർട്ട് വിശദീകരിക്കൂ"],
    ["पृष्ठ 3 पर क्या दिखाया गया है?"],
    ["ab mujhe btao ye paper kya kehna chahta hai?"],
]


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Multimodal Multilingual RAG") as demo:

        gr.Markdown("""
# Multimodal Multilingual RAG
Ask questions in **any language** about your PDF.
Upload below → wait for processing → start chatting!
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Step 1: Upload a PDF")
                pdf_upload = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                upload_btn = gr.Button("Process PDF", variant="primary")
                upload_status = gr.Textbox(
                    label="Status",
                    lines=5,
                    interactive=False,
                    placeholder="Upload a PDF and click Process PDF...",
                )

        gr.Markdown("---")
        gr.Markdown("### Step 2: Ask questions in any language")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=420)
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your question",
                        placeholder="Ask anything... / ചോദിക്കൂ... / पूछें... / Hinglish mein bhi!",
                        scale=4, lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(examples=EXAMPLES, inputs=msg_input, label="Example queries")

            with gr.Column(scale=2):
                with gr.Accordion("Search settings", open=False):
                    modality_filter = gr.Dropdown(
                        choices=["All", "Text", "Table", "Image"],
                        value="All", label="Filter by modality",
                    )
                    top_k_slider = gr.Slider(minimum=1, maximum=12, value=6, step=1, label="Top-k chunks")

                gr.Markdown("### Retrieved images")
                gallery = gr.Gallery(columns=2, height=300, show_label=False)

                gr.Markdown("### Sources")
                sources_md = gr.Markdown("_Sources appear here after first query._")

        gr.Button("Clear conversation").click(
            fn=lambda: ([], "", [], "_No sources._"),
            outputs=[chatbot, msg_input, gallery, sources_md],
        )

        gr.Markdown(f"""
---
**Stack:** multilingual-e5-large · {LLM_MODEL} via Ollama · Qdrant · PyMuPDF · LLaVA
**Languages:** English, Malayalam, Hindi, Tamil, Hinglish, Arabic + 100 more
        """)

        upload_btn.click(
            fn=process_uploaded_pdf,
            inputs=[pdf_upload],
            outputs=[upload_status],
        )
        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot, modality_filter, top_k_slider],
            outputs=[chatbot, msg_input, gallery, sources_md],
        )
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot, modality_filter, top_k_slider],
            outputs=[chatbot, msg_input, gallery, sources_md],
        )

    return demo


if __name__ == "__main__":
    print("Starting Multimodal Multilingual RAG UI...")
    print("Open: XX\n")
    build_ui().launch(server_name="X", server_port=X, share=False)