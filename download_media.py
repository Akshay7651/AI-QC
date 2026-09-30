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
