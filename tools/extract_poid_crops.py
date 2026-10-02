#!/usr/bin/env python3
"""Cut the handwritten PO ID strip of every Haryana form in data/forms -> data/poid_crops.pkl (list of dict docket, crop(grey uint8)).
The box from the layout is widened (to the right page edge, and up/down) because the handwriting often runs past it.
Resumable: forms already in the pickle are skipped.   python tools/extract_poid_crops.py [--limit N] [--procs 3]"""
import argparse
import glob
import os
import pickle
import random
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("OMP_THREAD_LIMIT", "1")
OUT = os.path.join(ROOT, "data", "poid_crops.pkl")


def one(path):
    import form_p3 as P
    import poid_reader as PR
    d = os.path.basename(path)[:-4]
    try:
        lay = P.analyse_layout(P.load_image(path))
        if not lay.get("ok") or "po_id" not in (lay.get("boxes") or {}):
            return d, None
        box, h = PR.window_box(lay)
        return d, PR.find_line(P.crop(lay["rgb"], box), h)
    except Exception:     # noqa: BLE001
        return d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(ROOT, "data", "forms", "*.jpg")))
    random.Random(7).shuffle(fs)
    fs = fs[: a.limit]
    have = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else []
    done = {x["docket"] for x in have}
    todo = [f for f in fs if os.path.basename(f)[:-4] not in done]
    print(f"have {len(have)} | to do {len(todo)}", flush=True)
    from concurrent.futures import ProcessPoolExecutor
    n = 0
    with ProcessPoolExecutor(a.procs) as ex:
        for d, g in ex.map(one, todo, chunksize=8):
            n += 1
            if g is not None:
                have.append({"docket": d, "crop": g})
            if n % 200 == 0 or n == len(todo):
                tmp = OUT + ".tmp"
                pickle.dump(have, open(tmp, "wb"))
                os.replace(tmp, OUT)
                print(f"{n}/{len(todo)} processed | crops {len(have)}", flush=True)


if __name__ == "__main__":
    main()
