#!/usr/bin/env python3
import os, re, io, argparse
from pathlib import Path
from PyPDF2 import PdfMerger, PdfReader, PdfWriter

# --- filename parsing for ordering ---
WEEK_RE = re.compile(r"week[\s._-]?(\d+)", re.IGNORECASE)
LEC_RE  = re.compile(r"(?:lec(?:ture)?|day)[\s._-]?(\d+)", re.IGNORECASE)

def week_sort_key(path: Path):
    s = path.as_posix()
    w = WEEK_RE.search(s)
    l = LEC_RE.search(s)
    return (int(w.group(1)) if w else 10_000,
            int(l.group(1)) if l else 10_000,
            s.lower())

def find_pdf_files(root: Path, output_file: Path):
    pdfs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                p = Path(dirpath) / fn
                if p.resolve() != output_file.resolve():
                    pdfs.append(p)
    return pdfs

# --- stamping helpers (Option 1) ---
def _make_number_overlay(width, height, text, x="center", y_margin=36, font="Helvetica", size=10):
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(width, height))
    c.setFont(font, size)
    c.setFillGray(0.25)  # subtle grey
    # position
    tw = c.stringWidth(text, font, size)
    if x == "center":
        x_pos = (width - tw) / 2.0
    elif x == "right":
        x_pos = width - (0.5 * inch) - tw
    else:
        x_pos = 0.5 * inch
    y_pos = y_margin  # from bottom
    c.drawString(x_pos, y_pos, text)
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]

def stamp_page_numbers(reader: PdfReader) -> PdfWriter:
    writer = PdfWriter()
    total = len(reader.pages)
    for i, page in enumerate(reader.pages, start=1):
        w = float(page.mediabox.width)
        h = float(page.mediabox.height)
        overlay = _make_number_overlay(w, h, f"Page {i} of {total}")
        # PyPDF2 v3: merge_page; older: mergePage
        if hasattr(page, "merge_page"):
            page.merge_page(overlay)
        else:
            page.mergePage(overlay)
        writer.add_page(page)
    return writer

# --- labels-only helpers (Option 2) ---
def add_page_labels(reader: PdfReader) -> PdfWriter:
    # Uses pypdf's convenient API if available; otherwise, no-op fallback.
    try:
        from pypdf import PdfWriter as PPWriter
        pw = PPWriter()
        for p in reader.pages:
            pw.add_page(p)
        # make labels 1..N with decimal style
        pw.add_page_label(0, label_style="D", start=1)
        return pw
    except Exception:
        # Fallback: return unchanged (labels may not be added)
        w = PdfWriter()
        for p in reader.pages:
            w.add_page(p)
        return w

def main():
    ap = argparse.ArgumentParser(description="Combine lecture PDFs and add continuous page numbers.")
    ap.add_argument("-d", "--dir", default=".", help="Root directory to search (default: current dir).")
    ap.add_argument("-o", "--output", default="ALL_EE102_notes_compiled.pdf", help="Output PDF filename.")
    ap.add_argument("--labels-only", action="store_true",
                    help="Do not stamp content; set viewer page labels only.")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    output = (root / args.output).resolve()

    pdf_files = find_pdf_files(root, output)
    if not pdf_files:
        print(f"No PDFs found under: {root}")
        return
    pdf_files.sort(key=week_sort_key)

    # Merge first (in-memory)
    merged_buf = io.BytesIO()
    merger = PdfMerger(strict=False)
    for p in pdf_files:
        try:
            merger.append(str(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    merger.write(merged_buf)
    merger.close()
    merged_buf.seek(0)

    combined_reader = PdfReader(merged_buf)
    if len(combined_reader.pages) == 0:
        print("Merged file has 0 pages. Check inputs.")
        return

    if args.labels_only:
        writer = add_page_labels(combined_reader)
    else:
        writer = stamp_page_numbers(combined_reader)

    with open(output, "wb") as f:
        writer.write(f)

    print(f"Combined PDF saved to: {output}")
    print(f"Pages: {len(combined_reader.pages)}")
    if args.labels_only:
        print("Applied viewer page labels (content unchanged).")
    else:
        print("Stamped continuous page numbers at bottom center.")

if __name__ == "__main__":
    main()
