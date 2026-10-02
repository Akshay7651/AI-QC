#!/usr/bin/env python3
"""Convert every PDF in a folder to JPG images, named by the PDF name (= docket id).

  pip install pymupdf
  python tools/pdf_to_images.py --folder "D:\\All_Application\\AI-QC\\Input\\forms"
  python tools/pdf_to_images.py --folder forms --dpi 200 --delete-pdf

<docket>.pdf  ->  <docket>.jpg           (page 1)
                  <docket>_p2.jpg ...    (only if the PDF has more pages)
Resumable (already converted PDFs are skipped).  failed_pdf.csv lists every PDF that could not be converted, so nothing is dropped silently.
PDFs are kept unless you pass --delete-pdf (the PDF is deleted only after its JPG was written and re-opened successfully).
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import pymupdf
except ImportError:                 # older PyMuPDF
    import fitz as pymupdf


def convert(job):
    pdf, dpi, quality, delete = job
    pdf = Path(pdf)
    first = pdf.with_suffix(".jpg")
    try:
        if first.exists() and first.stat().st_size > 0:
            return pdf.name, "skipped", ""
        with pymupdf.open(pdf) as doc:
            n = doc.page_count
            if n == 0:
                return pdf.name, "failed", "PDF has no pages"
            outs = []
            for i in range(n):
                pix = doc[i].get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
                out = first if i == 0 else pdf.with_name(f"{pdf.stem}_p{i + 1}.jpg")
                pix.save(out, jpg_quality=quality)
                outs.append(out)
        for o in outs:                                        # verify before anything is deleted
            with pymupdf.open(o) as chk:
                if chk.page_count < 1:
                    raise RuntimeError("written image cannot be re-opened")
        if delete:
            pdf.unlink()
        return pdf.name, "ok", f"{n} page(s)"
    except Exception as e:     # noqa: BLE001
        return pdf.name, "failed", f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--quality", type=int, default=90, help="JPG quality 1-100")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--delete-pdf", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N PDFs (to test)")
    a = ap.parse_args()
    folder = Path(a.folder)
    pdfs = sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))
    pdfs = pdfs[: a.limit] if a.limit else pdfs
    if not pdfs:
        sys.exit(f"no PDF files in {folder}")
    print(f"{len(pdfs)} PDFs", flush=True)
    jobs = [(str(p), a.dpi, a.quality, a.delete_pdf) for p in pdfs]
    ok = skipped = 0
    failed = []
    with ProcessPoolExecutor(a.procs) as ex:
        for i, (name, status, msg) in enumerate(ex.map(convert, jobs, chunksize=4), 1):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed.append((name, msg))
            if i % 100 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} | converted {ok} | already done {skipped} | failed {len(failed)}", flush=True)
    with open(folder / "failed_pdf.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pdf", "reason"])
        w.writerows(failed)
    print(f"done. converted {ok}, already done {skipped}, failed {len(failed)} (see {folder / 'failed_pdf.csv'})")


if __name__ == "__main__":
    main()
