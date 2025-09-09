#!/usr/bin/env python3
import os
import re
from pathlib import Path
from PyPDF2 import PdfMerger
import argparse

WEEK_RE = re.compile(r"week[\s_-]?(\d+)", re.IGNORECASE)
LEC_RE  = re.compile(r"(?:lec(?:ture)?|day)[\s_-]?(\d+)", re.IGNORECASE)

def week_sort_key(path: Path):
    s = path.as_posix()
    w = WEEK_RE.search(s)
    l = LEC_RE.search(s)
    week = int(w.group(1)) if w else 10_000
    lec  = int(l.group(1)) if l else 10_000
    return (week, lec, s.lower())

def find_pdf_files(root: Path, output_file: Path):
    pdfs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if not fn.lower().endswith(".pdf"):
                continue
            p = Path(dirpath) / fn
            # don't include the output file in case of re-runs
            if p.resolve() == output_file.resolve():
                continue
            pdfs.append(p)
    return pdfs

def main():
    ap = argparse.ArgumentParser(description="Combine lecture note PDFs into one file.")
    ap.add_argument("-d", "--dir", default=".", help="Root directory to search (default: current dir).")
    ap.add_argument("-o", "--output", default="ALL_EE102_notes_compiled.pdf", help="Output PDF filename.")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    output = (root / args.output).resolve()

    pdf_files = find_pdf_files(root, output)
    if not pdf_files:
        print(f"No PDFs found under: {root}")
        return

    pdf_files.sort(key=week_sort_key)

    print("Merging PDFs in this order:")
    for i, p in enumerate(pdf_files, 1):
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        print(f"{i:02d}. {rel}")

    merger = PdfMerger(strict=False)
    for p in pdf_files:
        try:
            merger.append(str(p))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    merger.write(str(output))
    merger.close()
    print(f"\nCombined PDF saved to: {output}")

if __name__ == "__main__":
    main()
