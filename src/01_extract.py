"""
01_extract.py — Step 1: Extract text, tables, and images from PDFs.

Run: python src/01_extract.py

What it does:
- Reads every PDF in data/pdfs/
- Extracts text blocks page by page
- Extracts tables as CSV rows
- Extracts embedded images (skips tiny icons)
- Saves everything to data/extracted/{text,tables,images}/
"""

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

# Add project root to path so config import works
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PDF_DIR, TEXT_DIR, TABLE_DIR, IMAGE_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT, IMAGE_DPI
)


# ── Text chunking helper ─────────────────────────────────────────────────────

def chunk_text(text: str, source: str, page: int) -> list[dict]:
    """Split a long text string into overlapping chunks with metadata."""
    chunks = []
    start = 0
    chunk_idx = 0
    text = text.strip()
    if not text:
        return []

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({
            "text":       chunk,
            "source":     source,
            "page":       page,
            "chunk_idx":  chunk_idx,
            "modality":   "text",
        })
        chunk_idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ── Table extraction helper ──────────────────────────────────────────────────

def extract_tables_from_page(page, source: str, page_num: int) -> list[dict]:
    """
    PyMuPDF can find table structures on a page.
    Each table row becomes a separate chunk with modality='table'.
    """
    tables = []
    try:
        found = page.find_tables()
        for table_idx, table in enumerate(found.tables):
            df = table.to_pandas()
            # Convert each row to a readable text string
            for row_idx, row in df.iterrows():
                row_text = " | ".join(str(v) for v in row.values if str(v).strip())
                if row_text.strip():
                    tables.append({
                        "text":      row_text,
                        "source":    source,
                        "page":      page_num,
                        "table_idx": table_idx,
                        "row_idx":   int(row_idx),
                        "modality":  "table",
                    })
    except Exception as e:
        print(f"  [warn] table extraction failed on page {page_num}: {e}")
    return tables


# ── Image extraction helper ──────────────────────────────────────────────────

def extract_images_from_page(doc, page, source_stem: str, page_num: int) -> list[dict]:
    """
    Extract embedded images from a PDF page.
    Saves each image as PNG, returns metadata list.
    """
    images = []
    img_list = page.get_images(full=True)

    for img_idx, img_info in enumerate(img_list):
        xref = img_info[0]
        try:
            base_image = doc.extract_image(xref)
            img_bytes  = base_image["image"]
            img_ext    = base_image["ext"]

            # Load with fitz to check dimensions
            img_rect = fitz.Pixmap(doc, xref)

            # Skip tiny images (icons, decorations)
            if img_rect.width < MIN_IMAGE_WIDTH or img_rect.height < MIN_IMAGE_HEIGHT:
                img_rect = None
                continue

            # Save image to disk
            img_filename = f"{source_stem}_page{page_num:03d}_img{img_idx:02d}.png"
            img_path = IMAGE_DIR / img_filename

            # Convert to PNG regardless of source format
            if img_rect.n >= 5:  # CMYK — convert to RGB first
                img_rect = fitz.Pixmap(fitz.csRGB, img_rect)
            img_rect.save(str(img_path))

            images.append({
                "image_path": str(img_path),
                "source":     source_stem,
                "page":       page_num,
                "img_idx":    img_idx,
                "width":      img_rect.width,
                "height":     img_rect.height,
                "modality":   "image",
                # caption will be filled in by 02_caption.py
                "caption":    None,
            })

            img_rect = None

        except Exception as e:
            print(f"  [warn] image {img_idx} on page {page_num} failed: {e}")

    return images


# ── Main extraction loop ─────────────────────────────────────────────────────

def process_pdf(pdf_path: Path):
    print(f"\nProcessing: {pdf_path.name}")
    source_stem = pdf_path.stem  # filename without extension

    doc = fitz.open(str(pdf_path))
    all_text   = []
    all_tables = []
    all_images = []

    for page_num in tqdm(range(len(doc)), desc="  Pages", leave=False):
        page = doc[page_num]

        # 1. Extract text
        text = page.get_text("text")
        chunks = chunk_text(text, source_stem, page_num + 1)
        all_text.extend(chunks)

        # 2. Extract tables
        tables = extract_tables_from_page(page, source_stem, page_num + 1)
        all_tables.extend(tables)

        # 3. Extract images
        images = extract_images_from_page(doc, page, source_stem, page_num + 1)
        all_images.extend(images)

    doc.close()

    # Save outputs as JSON
    text_out   = TEXT_DIR  / f"{source_stem}_text.json"
    table_out  = TABLE_DIR / f"{source_stem}_tables.json"
    image_out  = IMAGE_DIR / f"{source_stem}_images.json"
    text_out.write_text(json.dumps(all_text,   indent=2, ensure_ascii=True), encoding='utf-8')
    table_out.write_text(json.dumps(all_tables, indent=2, ensure_ascii=True), encoding='utf-8')
    image_out.write_text(json.dumps(all_images, indent=2, ensure_ascii=True), encoding='utf-8') 
    print(f"  Done: {len(all_text)} text chunks, {len(all_tables)} table rows, {len(all_images)} images")
    return len(all_text), len(all_tables), len(all_images)


def main():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}")
        print("Add some PDF files to data/pdfs/ and run again.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process")
    total_text = total_tables = total_images = 0

    for pdf_path in pdf_files:
        t, tb, i = process_pdf(pdf_path)
        total_text   += t
        total_tables += tb
        total_images += i

    print(f"\nExtraction complete!")
    print(f"  Total text chunks : {total_text}")
    print(f"  Total table rows  : {total_tables}")
    print(f"  Total images      : {total_images}")
    print(f"\nNext step: python src/02_caption.py")


if __name__ == "__main__":
    main()
