#!/usr/bin/env python3
"""Retrain the handwritten-cell model from human-QC'd rows, safely.

  python retrain.py --labels human_qc.xlsx [--steps 6000] [--holdout 0.15] [--dry-run]

Needs a human-QC'd Excel with: Docket_ID, Signed_Copy_URL (form link) and the human columns
'Affected area% (Form)' and 'Crop Loss% (Form)'.  What it does:
  1. reads the human values, splits dockets 85/15 (the 15% are NEVER trained on -> honest test)
  2. downloads the forms it does not have yet into data/forms/ (polite, resumable)
  3. builds the cell-crop cache, measures the CURRENT model on the human hold-out
  4. trains the new model (human values override the app values)
  5. measures the new model on the same hold-out and ADOPTS it only if it is not worse; otherwise restores the backup.
"""
import argparse
import csv
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "cells_cnn.npz"
BACKUP_DIR = HERE / "models" / "backup"
GATE = 0.8


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def find_col(df, *names):
    want = {norm(n) for n in names}
    for c in df.columns:                      # exact match first, so 'Affected_Area_%' wins over 'Affected area% (Form)'
        if norm(c) in want:
            return c
    for c in df.columns:
        if any(norm(c).startswith(w) for w in want):
            return c
    return None


def load_app_values(path, max_rows=3000, zero_share=0.3, seed=7):
    """Weak labels straight from the CLAP export: the values the surveyor typed into the app (columns pdf_url, affected_area_pct,
    crop_loss_pct).  The reader is weakest on NON-ZERO handwriting, so every non-zero row is taken first and zero rows are limited
    to zero_share of the sample.  The hold-out is still never trained on, and the new model is adopted only if it is not worse."""
    df = pd.read_excel(path, dtype=str) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
    d = find_col(df, "Docket_ID", "docketID", "docket")
    url = find_col(df, "pdf_url", "Signed_Copy_URL", "form_url")
    a = find_col(df, "affected_area_pct", "Affected_Area_%")
    l = find_col(df, "crop_loss_pct", "Crop_Loss_%")
    if None in (d, url, a, l):
        sys.exit("Need columns docket_id, pdf_url, affected_area_pct, crop_loss_pct. Found: " + ", ".join(map(str, df.columns)))
    out = pd.DataFrame({"docket_id": df[d].astype(str).str.strip(), "pdf_url": df[url],
                        "affected_area_pct": pd.to_numeric(df[a].astype(str).str.replace("%", "").str.strip(), errors="coerce"),
                        "crop_loss_pct": pd.to_numeric(df[l].astype(str).str.replace("%", "").str.strip(), errors="coerce")})
    out = out.dropna(subset=["affected_area_pct", "crop_loss_pct"]).drop_duplicates("docket_id")
    out = out[out.pdf_url.notna() & out.pdf_url.astype(str).str.startswith("http")]
    ok5 = lambda s: (s % 5 == 0) & (s >= 0) & (s <= 100)
    out = out[ok5(out.affected_area_pct) & ok5(out.crop_loss_pct)]          # the cell model knows multiples of 5 only
    nz = out[(out.affected_area_pct > 0) | (out.crop_loss_pct > 0)]
    z = out.drop(nz.index)
    nz = nz.sample(frac=1, random_state=seed).head(max_rows)
    z = z.sample(frac=1, random_state=seed).head(int(len(nz) * zero_share / max(1e-9, 1 - zero_share)) + 1)
    print(f"app-value labels: {len(nz)} non-zero rows + {len(z)} zero rows")
    return pd.concat([nz, z]).reset_index(drop=True)


def load_human(path):
    df = pd.read_excel(path, dtype=str) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
    d = find_col(df, "Docket_ID", "docketID", "docket")
    url = find_col(df, "Signed_Copy_URL", "pdf_url", "form_url")
    a = find_col(df, "Affected area% (Form)", "Affected_area_form")
    l = find_col(df, "Crop Loss% (Form)", "Crop_loss_form")
    missing = [n for n, c in (("Docket_ID", d), ("Signed_Copy_URL", url), ("Affected area% (Form)", a), ("Crop Loss% (Form)", l)) if c is None]
    if missing:
        sys.exit("Missing columns in the human-QC file: " + ", ".join(missing) + "\nFound: " + ", ".join(map(str, df.columns)))
    out = pd.DataFrame({"docket_id": df[d].astype(str).str.strip(), "pdf_url": df[url],
                        "affected_area_pct": pd.to_numeric(df[a].astype(str).str.replace("%", "").str.strip(), errors="coerce"),
                        "crop_loss_pct": pd.to_numeric(df[l].astype(str).str.replace("%", "").str.strip(), errors="coerce")})
    out = out.dropna(subset=["affected_area_pct", "crop_loss_pct"], how="all").drop_duplicates("docket_id")
    return out[out.pdf_url.notna()]


def download_forms(df, threads=8):
    import httpx
    import fetch_dataset as F
    forms = HERE / "data" / "forms"
    forms.mkdir(parents=True, exist_ok=True)
    todo = [r for _, r in df.iterrows() if not (forms / f"{r.docket_id}.jpg").exists()]
    print(f"forms to download: {len(todo)} (already have {len(df) - len(todo)})", flush=True)
    done = [0]
    lock = threading.Lock()

    def one(r):
        with httpx.Client() as c:
            b = F.get(c, str(r.pdf_url).replace("_x000D_", "").strip())
            if b:
                try:
                    (forms / f"{r.docket_id}.jpg").write_bytes(F.pdf_to_jpg(b))
                except Exception:
                    pass
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  downloaded {done[0]}/{len(todo)}", flush=True)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(one, todo))


def measure(dockets, human):
    """precision-at-gate and coverage of the CURRENT models/cells_cnn.npz on cached crops of the given dockets."""
    import importlib
    import digits as D
    importlib.reload(D)
    cache = HERE / "data" / "digits_cache"
    n = ans = ok = 0
    for d in dockets:
        f = cache / f"{d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        if not int(z["ok"]):
            continue
        for key, col in (("area", "affected_area_pct"), ("loss", "crop_loss_pct")):
            v = human.loc[d, col]
            if pd.isna(v) or z[key].size < 500:
                continue
            n += 1
            res = D.read_number(z[key], gate=GATE)
            if res.get("value") is not None:
                ans += 1
                ok += int(abs(res["value"] - v) < 1e-6)
    return {"cells": n, "answered": ans, "correct": ok, "coverage": ans / max(n, 1), "precision": ok / max(ans, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="human-QC'd Excel/CSV")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--app-values", action="store_true", help="--labels is the raw CLAP export: train on the values typed in the app (weak labels)")
    ap.add_argument("--max-rows", type=int, default=3000, help="with --app-values: how many non-zero forms to use")
    ap.add_argument("--init", action="store_true", help="start from the current model instead of from scratch (fine-tune)")
    a = ap.parse_args()

    hum = load_app_values(a.labels, a.max_rows) if a.app_values else load_human(a.labels)
    print(f"human rows with a form link and a value: {len(hum)}")
    ids = sorted(hum.docket_id, key=lambda d: hashlib.md5(d.encode()).hexdigest())
    k = int(len(ids) * a.holdout)
    hold, train = set(ids[:k]), set(ids[k:])
    print(f"train {len(train)} | hold-out {len(hold)} (never trained on)")
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    hum.drop(columns=["pdf_url"]).to_csv(data / "manifest_human.csv", index=False)
    pd.DataFrame({"docket_id": sorted(hold)}).to_csv(data / "holdout_human.csv", index=False)
    if a.dry_run:
        print("dry run: wrote data/manifest_human.csv and data/holdout_human.csv; nothing downloaded or trained.")
        return
    if not a.no_download:
        download_forms(hum, a.threads)
    import train_digits as TD
    print("building cell-crop cache (skips forms already cached) ...", flush=True)
    TD.build_cache(procs=a.procs)
    h = hum.set_index("docket_id")
    old = measure(sorted(hold), h)
    print("CURRENT model on human hold-out:", old, flush=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bk = BACKUP_DIR / f"cells_cnn_{stamp}.npz"
    if MODEL.exists():
        shutil.copy2(MODEL, bk)
    print("training new model ...", flush=True)
    subprocess.run([sys.executable, str(HERE / "train_cells.py"), "--rebuild", "--steps", str(a.steps)] + (["--init"] if a.init else []), check=True, cwd=HERE)
    new = measure(sorted(hold), h)
    print("NEW model on human hold-out:", new, flush=True)
    shutil.copy2(MODEL, HERE / "models" / "cells_cnn_candidate.npz")      # kept for a separate comparison whatever is decided below
    adopt = new["correct"] >= old["correct"] and (new["precision"] >= 0.95 or new["precision"] >= old["precision"] - 0.01)
    if adopt:
        print(f"ADOPTED the new model (backup of the old one: {bk.name}).")
    else:
        if bk.exists():
            shutil.copy2(bk, MODEL)
        print("NOT adopted: the new model is worse on the hold-out; the previous model was restored.")
    print("Also run: python eval_digits.py   (scores on the hand-labelled forms)")


if __name__ == "__main__":
    main()
