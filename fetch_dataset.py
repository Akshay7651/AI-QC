#!/usr/bin/env python3
"""Resumable, polite bulk downloader of signed forms (and a sample of photos) for training/evaluation.

  python fetch_dataset.py --input Level_1.xlsx --n 12000 --photo-rows 1500 --out data
Saves the JPEG embedded in each form PDF untouched (no re-render) as data/forms/<docket>.jpg,
photos (downscaled) as data/photos/<docket>_<i>.jpg, and data/manifest.csv. Safe to re-run (skips done).
"""
import argparse
import csv
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pandas as pd
from PIL import Image

import ingestion
from photo_qc import split_urls

H = [{}, {"Referer": "https://pmfby.gov.in"}]
lock = threading.Lock()


def get(c, url):
    for attempt in range(3):
        for h in H:
            try:
                r = c.get(url, headers=h, timeout=40, follow_redirects=True)
                if r.status_code == 200 and r.content:
                    return r.content
            except httpx.HTTPError:
                pass
        time.sleep(2 * (attempt + 1))
    return None


def pdf_to_jpg(data: bytes):
    import pymupdf
    with pymupdf.open(stream=data, filetype="pdf") as d:
        imgs = d[0].get_images(full=True)
        if imgs:
            info = d.extract_image(imgs[0][0])
            if info["ext"] in ("jpeg", "jpg"):
                return info["image"]
        return d[0].get_pixmap(dpi=150).tobytes("jpg")


def photos_only(a, df, out):
    have = {p.stem.rsplit("_", 1)[0] for p in (out / "photos").glob("*.jpg")}
    forms = [p.stem for p in (out / "forms").glob("*.jpg")]
    df = df.set_index(df["docket_id"].astype(str), drop=False)
    todo = [d for d in forms if d not in have and d in df.index]
    print(f"{len(have)} rows already have photos; {len(todo)} candidates; target {a.photos_target} photos", flush=True)
    cnt = [len(list((out / "photos").glob("*.jpg")))]
    lk = threading.Lock()

    def one(d):
        if cnt[0] >= a.photos_target:
            return
        urls = split_urls(df.loc[d, "media_urls"])[:3]
        with httpx.Client() as c:
            for i, u in enumerate(urls, 1):
                b = get(c, u)
                if b:
                    try:
                        im = Image.open(io.BytesIO(b)).convert("RGB")
                        im.thumbnail((720, 1280))
                        im.save(out / "photos" / f"{d}_{i}.jpg", quality=82)
                        with lk:
                            cnt[0] += 1
                    except Exception:
                        pass
        if cnt[0] % 200 < 3:
            print(f"photos {cnt[0]}", flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(one, todo))
    print("PHOTOS DONE", cnt[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--photo-rows", type=int, default=1500)
    ap.add_argument("--out", default="data")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--photos-target", type=int, help="photos-only mode: add up to 3 photos for already-downloaded rows until this many photo files exist")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "forms").mkdir(parents=True, exist_ok=True)
    (out / "photos").mkdir(exist_ok=True)
    df = ingestion.load(a.input)
    if a.photos_target:
        return photos_only(a, df, out)
    df = df[df["pdf_url"].notna()]
    # stratified by district so every area/surveyor mix is represented
    frac = min(1.0, a.n / len(df))
    take = df.groupby("district").sample(frac=frac, random_state=a.seed)
    take = take.sample(frac=1, random_state=a.seed).head(a.n).reset_index(drop=True)
    photo_set = set(take["docket_id"].head(a.photo_rows))
    man = out / "manifest.csv"
    new = not man.exists()
    mf = open(man, "a", newline="")
    w = csv.writer(mf)
    if new:
        w.writerow(["docket_id", "district", "surveyor_name", "crop_name", "affected_area_pct", "crop_loss_pct",
                    "total_damage_pct", "latitude", "longitude", "survey_start_date", "survey_end_date", "form_ok", "n_photos"])
    done = {p.stem for p in (out / "forms").glob("*.jpg")}
    cnt = [0, 0]
    t0 = time.time()

    def work(r):
        d = str(r["docket_id"]).strip()
        fo, npho = d in done, 0
        with httpx.Client() as c:
            if not fo:
                data = get(c, str(r["pdf_url"]).strip())
                if data:
                    try:
                        (out / "forms" / f"{d}.jpg").write_bytes(pdf_to_jpg(data))
                        fo = True
                    except Exception:
                        pass
            if d in photo_set:
                for i, u in enumerate(split_urls(r["media_urls"])[:3], 1):
                    p = out / "photos" / f"{d}_{i}.jpg"
                    if p.exists():
                        npho += 1
                        continue
                    b = get(c, u)
                    if b:
                        try:
                            im = Image.open(io.BytesIO(b)).convert("RGB")
                            im.thumbnail((720, 1280))
                            im.save(p, quality=82)
                            npho += 1
                        except Exception:
                            pass
        with lock:
            cnt[0 if fo else 1] += 1
            w.writerow([d, r["district"], r["surveyor_name"], r["crop_name"], r["affected_area_pct"], r["crop_loss_pct"],
                        r["total_damage_pct"], r["latitude"], r["longitude"], r["survey_start_date"], r["survey_end_date"], int(fo), npho])
            mf.flush()
            if sum(cnt) % 100 == 0:
                print(f"{sum(cnt)}/{len(take)} ok={cnt[0]} fail={cnt[1]} {time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(work, [r for _, r in take.iterrows()]))
    print("DONE", cnt)


if __name__ == "__main__":
    main()
