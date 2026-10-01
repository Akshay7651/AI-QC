#!/usr/bin/env python3
"""Download the form PDF + photos from the links in a CLAP Excel into <out>/ named by docket id, then zip.

  python download_media.py --input test10.xlsx --out downloads --limit 10
Creates downloads/<docket>.pdf, downloads/<docket>_1.jpg ... and downloads.zip
Run this on a PC that can open the pmfby.gov.in links in a browser.
"""
import argparse
import shutil
import time
from pathlib import Path

import httpx

import ingestion
from photo_qc import split_urls

REFERER = {"Referer": "https://pmfby.gov.in"}


def ext_for(content: bytes, default: str) -> str:
    if content[:4] == b"%PDF":
        return ".pdf"
    if content[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if content[:4] == b"RIFF":
        return ".webp"
    return default


def get(client, url):
    for h in ({}, REFERER):
        try:
            r = client.get(url, headers=h, timeout=30, follow_redirects=True)
            if r.status_code == 200 and r.content:
                return r.content
        except httpx.HTTPError:
            pass
    return None


def download_all(df, out, threads=12, max_photos=10, log=print):
    """Download every row's signed form and photos to  <out>/<docket>/form/<docket>.<ext>  and  <out>/<docket>/media/<docket>_<n>.<ext>.
    Resumable (existing files are skipped) and parallel.  Returns (rows_with_form, rows_without_form, files_failed)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df.iterrows():
        d = str(r["docket_id"]).strip()
        if d and d.lower() != "nan":
            rows.append((d, str(r["pdf_url"] or "").replace("_x000D_", "").strip(), split_urls(r["media_urls"])[:max_photos]))
    st = {"done": 0, "fail": 0, "noform": 0, "bytes": 0, "t0": time.time()}
    lock = threading.Lock()

    def have(folder, stem):
        return folder.exists() and any(f.stem == stem and f.stat().st_size > 0 for f in folder.iterdir())

    def one(job):
        d, pdf, photos = job
        fo, me = out / d / "form", out / d / "media"
        fo.mkdir(parents=True, exist_ok=True)
        me.mkdir(parents=True, exist_ok=True)
        todo = ([(fo, d, pdf, ".pdf")] if pdf.startswith("http") else []) + [(me, f"{d}_{i}", u, ".jpg") for i, u in enumerate(photos, 1) if u]
        bad = 0
        with httpx.Client() as c:
            for folder, stem, url, dflt in todo:
                if have(folder, stem):
                    continue
                data = get(c, url)
                if data:
                    with lock:
                        st["bytes"] += len(data)
                    ext = ext_for(data, dflt)
                    if ext == ".png" and folder.name == "media":       # photos come as ~5 MB PNG: store as JPEG (about 10x smaller on disk)
                        try:
                            import io
                            from PIL import Image
                            buf = io.BytesIO()
                            Image.open(io.BytesIO(data)).convert("RGB").save(buf, "JPEG", quality=92)
                            data, ext = buf.getvalue(), ".jpg"
                        except Exception:
                            pass
                    (folder / f"{stem}{ext}").write_bytes(data)
                else:
                    bad += 1
        with lock:
            st["done"] += 1
            st["fail"] += bad
            st["noform"] += 0 if pdf.startswith("http") else 1
            n = st["done"]
            if n % 10 == 0 or n == len(rows):
                el = time.time() - st["t0"]
                log(f"  downloaded {n}/{len(rows)} rows  ({n / max(el, 1e-6) * 60:.0f} rows/min, {st['bytes'] / max(el, 1e-6) / 1048576:.1f} MB/s from the server, {st['fail']} files failed)")

    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, rows))
    return len(rows) - st["noform"], st["noform"], st["fail"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="downloads")
    ap.add_argument("--limit", type=int, help="only the first N rows")
    a = ap.parse_args()
    df = ingestion.load(a.input)
    if a.limit:
        df = df.head(a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    with httpx.Client() as c:
        for _, r in df.iterrows():
            d = str(r["docket_id"]).strip()
            jobs = []
            if r["pdf_url"]:
                jobs.append((d, str(r["pdf_url"]).replace("_x000D_", "").strip(), ".pdf"))
            for i, u in enumerate(split_urls(r["media_urls"])[:5], 1):
                jobs.append((f"{d}_{i}", u, ".jpg"))
            for name, url, dflt in jobs:
                data = get(c, url)
                if data:
                    (out / f"{name}{ext_for(data, dflt)}").write_bytes(data)
                    ok += 1
                else:
                    fail += 1
                    print("FAILED:", name, url[:90])
                time.sleep(0.2)
    shutil.make_archive(str(out), "zip", out)
    print(f"Downloaded {ok}, failed {fail}. ZIP: {out}.zip")


if __name__ == "__main__":
    main()
