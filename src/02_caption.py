"""
02_caption.py — Step 2: Caption every extracted image using LLaVA.

Run: python src/02_caption.py

What it does:
- Reads all image metadata from data/extracted/images/*.json
- Sends each image to LLaVA (running locally via Ollama)
- LLaVA describes charts, diagrams, tables-as-images in plain text
- Saves updated metadata with captions to data/extracted/captions/

Prerequisites:
  ollama pull llava     ← run this once before using this script
"""

import base64
import json
import sys
import time
from pathlib import Path

import ollama
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import IMAGE_DIR, CAPTION_DIR, VISION_MODEL


# ── Prompt sent to LLaVA for every image ────────────────────────────────────

CAPTION_PROMPT = """You are analyzing an image extracted from a document.
Describe what you see in detail. If this is:
- A chart or graph: describe the title, axes, data trends, and key values
- A table: describe the columns, rows, and key data points
- A diagram or figure: describe what it shows and its components
- A photo or illustration: describe the content

Be specific and thorough. Your description will be used to answer questions about this image.
Respond in English only."""


def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_image(image_path: str, retries: int = 2) -> str:
    """
    Send an image to LLaVA via Ollama and get a text description.
    Retries on failure.
    """
    for attempt in range(retries + 1):
        try:
            response = ollama.chat(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": CAPTION_PROMPT,
                        "images": [image_path],  # Ollama accepts file paths directly
                    }
                ],
            )
            return response["message"]["content"].strip()

        except Exception as e:
            if attempt < retries:
                print(f"  [retry {attempt+1}] {e}")
                time.sleep(2)
            else:
                print(f"  [failed] Could not caption {image_path}: {e}")
                return f"[Caption failed: {e}]"


def check_ollama_running():
    """Check that Ollama is running and LLaVA is available."""
    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
        llava_available = any(VISION_MODEL in name for name in model_names)
        if not llava_available:
            print(f"ERROR: '{VISION_MODEL}' model not found in Ollama.")
            print(f"Run: ollama pull {VISION_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama at localhost:11434")
        print(f"Make sure Ollama is running: https://ollama.com/download")
        print(f"Details: {e}")
        return False


def main():
    if not check_ollama_running():
        sys.exit(1)

    # Find all image metadata files
    image_json_files = list(IMAGE_DIR.glob("*_images.json"))
    if not image_json_files:
        print(f"No image metadata found in {IMAGE_DIR}")
        print("Run python src/01_extract.py first.")
        return

    total_captioned = 0
    total_failed = 0

    for json_file in image_json_files:
        source_stem = json_file.stem.replace("_images", "")
        print(f"\nCaptioning images from: {source_stem}")

        images = json.loads(json_file.read_text())
        if not images:
            print("  No images to caption.")
            continue

        for item in tqdm(images, desc="  Images"):
            img_path = item.get("image_path")

            # Skip if already captioned
            if item.get("caption") and not item["caption"].startswith("[Caption failed"):
                continue

            if not img_path or not Path(img_path).exists():
                item["caption"] = "[Image file not found]"
                total_failed += 1
                continue

            caption = caption_image(img_path)
            item["caption"] = caption

            if caption.startswith("[Caption failed"):
                total_failed += 1
            else:
                total_captioned += 1

        # Save updated metadata with captions
        caption_out = CAPTION_DIR / f"{source_stem}_captions.json"
        caption_out.write_text(json.dumps(images, indent=2, ensure_ascii=True), encoding='utf-8')
        print(f"  Saved captions to {caption_out.name}")

    print(f"\nCaptioning complete!")
    print(f"  Captioned : {total_captioned}")
    print(f"  Failed    : {total_failed}")
    print(f"\nNext step: python src/03_embed_store.py")


if __name__ == "__main__":
    main()
