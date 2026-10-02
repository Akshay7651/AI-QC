#!/usr/bin/env python3
"""Download the signed form (and optionally the photos) of every row of a QC Excel, named by docket id.

  pip install pandas openpyxl httpx
  python tools/download_forms.py --excel Rajasthan_QC.xlsx --out forms
  python tools/download_forms.py --excel Rajasthan_QC.xlsx --out forms --photos      # also <docket>_1.jpg, <docket>_2.jpg ...

Result:  forms/<docket>.jpg (or .pdf / .png - the real file type is detected)   photos: forms/photos/<docket>_<n>.jpg
Resumable: files that already exist are skipped, so just run it again after an interruption.  failed.csv lists every docket that
could not be downloaded (with the reason), so nothing is silently dropped.
Columns used (change with the options if your Excel differs):  Docket_ID (a leading ' is removed),  Signed_Copy_URL (form),  Media (photo links, separated by , ; or spaces).
Run it on a PC/network that can open the pmfby.gov.in links in a browser.
"""
import argparse
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd

HEADERS = [{}, {"Referer": "https://pmfby.gov.in"}, {"Referer": "https://pmfby.gov.in", "User-Agent": "Mozilla/5.0"}]


def kind(content):
    if content[:4] == b"%PDF":
        return ".pdf"
    if content[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if content[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if content[:4] == b"RIFF":
        return ".webp"
    return None


def fetch(client, url):
    why = "no response"
    for h in HEADERS:
        try:
            r = client.get(url, headers=h, timeout=40, follow_redirects=True)
            if r.status_code == 200 and r.content:
                if kind(r.content):
                    return r.content, ""
                why = "not an image/PDF (login page or expired link?)"
            else:
                why = f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            why = type(e).__name__
    return None, why


def links(cell):
    s = str(cell or "").replace("_x000D_", " ")
    return [u for u in re.split(r"[\s,;|]+", s) if u.startswith("http")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True)
    ap.add_argument("--out", default="forms")
    ap.add_argument("--docket-col", default="Docket_ID")
    ap.add_argument("--form-col", default="Signed_Copy_URL")
    ap.add_argument("--media-col", default="Media")
    ap.add_argument("--dry-run", action="store_true", help="only read the Excel and print how many forms would be downloaded")
    ap.add_argument("--photos", action="store_true", help="also download the photos")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N rows (to test)")
    a = ap.parse_args()

    df = pd.read_excel(a.excel, dtype=str) if a.excel.lower().endswith((".xlsx", ".xlsm", ".xls")) else pd.read_csv(a.excel, dtype=str)
    low = {str(c).strip().lower(): c for c in df.columns}      # column names are matched ignoring upper/lower case
    a.docket_col, a.form_col, a.media_col = (low.get(x.lower(), x) for x in (a.docket_col, a.form_col, a.media_col))
    for c in (a.docket_col, a.form_col):
        if c not in df.columns:
            sys.exit(f"column '{c}' not found. Columns are: {', '.join(map(str, df.columns))}  (use --docket-col / --form-col)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(exist_ok=True)
    jobs = []
    for _, r in df.iterrows():
        d = str(r[a.docket_col]).strip().lstrip("'").strip()      # Excel text cells often start with a '
        if not d or d.lower() == "nan":
            continue
        jobs.append((d, links(r[a.form_col]), links(r.get(a.media_col)) if a.photos else []))
    jobs = jobs[: a.limit] if a.limit else jobs
    print(f"{len(jobs)} dockets | with a form link: {sum(1 for j in jobs if j[1])} | without: {sum(1 for j in jobs if not j[1])}", flush=True)
    if a.dry_run:
        return

    lock = threading.Lock()
    st = {"done": 0, "ok": 0, "skipped": 0, "failed": 0, "t0": time.time()}
    failed = []

    def have(folder, stem):
        return any(f.stem == stem and f.stat().st_size > 0 for f in folder.glob(stem + ".*"))

    def one(job):
        d, form, photos = job
        todo = []
        if form:
            todo.append((out, d, form[0]))
        else:
            with lock:
                failed.append((d, "form", "", "no form link in the Excel"))
        todo += [(out / "photos", f"{d}_{i}", u) for i, u in enumerate(photos, 1)]
        with httpx.Client() as c:
            for folder, stem, url in todo:
                if have(folder, stem):
                    with lock:
                        st["skipped"] += 1
                    continue
                data, why = fetch(c, url)
                if data is None:
                    with lock:
                        st["failed"] += 1
                        failed.append((d, stem, url, why))
                    continue
                (folder / (stem + kind(data))).write_bytes(data)
                with lock:
                    st["ok"] += 1
        with lock:
            st["done"] += 1
            if st["done"] % 50 == 0 or st["done"] == len(jobs):
                print(f"{st['done']}/{len(jobs)} dockets | downloaded {st['ok']} | already had {st['skipped']} | failed {st['failed']} "
                      f"| {time.time() - st['t0']:.0f}s", flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, jobs))
    pd.DataFrame(failed, columns=["docket", "file", "url", "reason"]).to_csv(out / "failed.csv", index=False)
    print(f"done. downloaded {st['ok']}, already had {st['skipped']}, failed {len(failed)} (see {out / 'failed.csv'})")


if __name__ == "__main__":
    main()
