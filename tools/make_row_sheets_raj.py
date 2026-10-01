#!/usr/bin/env python3
"""Contact sheets of the value table of Rajasthan forms, for hand labelling which ROW belongs to a docket and what it says.
  python tools/make_row_sheets_raj.py --labels raj_qc.csv --out data/eval_raj/row_sheets --per-sheet 3
Each strip: header + rows 1..10 of the table, with the TARGET docket tail printed above it.  index.csv: sheet,pos,docket.
Agents write data/eval_raj/rows_<k>.csv: docket,row,area,loss  (row 1..10 or ?, area/loss 0..100 or ?).
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def strip(path):
    import form_p3 as P
    import form_raj as R
    img = P.load_image(str(path))
    lay = P.analyse_layout(img)
    if not lay["ok"]:
        return None
    r = lay["rgb"]
    if not R.is_rajasthan(r):
        r2 = np.ascontiguousarray(r[::-1, ::-1])
        if not R.is_rajasthan(r2):
            return None
        lay = P._layout(r2, dict(lay.get("info") or {}))
        if not lay["ok"]:
            return None
    geo = R.rows_geometry(lay)
    if not geo:
        return None
    y0, p = geo
    sc = lay["scale"]
    hl, W = lay["hl"], lay["g"].shape[1]
    ext = [P.line_extent(hl, y0 + k * p, 0.012 * W) for k in range(11)]
    ext = [e for e in ext if e and e[1] - e[0] > 0.3 * W]
    if len(ext) < 2:
        return None
    L, Rr = float(np.median([e[0] for e in ext])), float(np.median([e[1] for e in ext]))
    if Rr - L < 0.8 * W:                  # the ruled lines may fade on the right: show everything up to the photo edge
        Rr = float(W)
    else:
        Rr = min(float(W), Rr + 0.1 * (Rr - L))      # the extent of the ruled lines can end a little before the real border
    ya, yb = (y0 - 2.5 * p) * sc, (y0 + 10.2 * p) * sc
    rgb = lay["rgb"]
    s = rgb[int(max(0, ya)): int(min(rgb.shape[0], yb)), int(max(0, L * sc - 10)): int(min(rgb.shape[1], Rr * sc + 10))]
    if s.size == 0:
        return None
    return cv2.resize(s, (1100, max(80, int(s.shape[0] * 1100 / s.shape[1]))))


def one(a):
    d, f = a
    try:
        return d, strip(f)
    except Exception:     # noqa: BLE001
        return d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--forms", default=str(ROOT / "data/forms_raj"))
    ap.add_argument("--out", default=str(ROOT / "data/eval_raj/row_sheets"))
    ap.add_argument("--per-sheet", type=int, default=3)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_csv(a.labels, dtype=str)
    docs = [str(d).strip() for d in df["docket_id" if "docket_id" in df else "docketID"]]
    forms = Path(a.forms)
    jobs = [(d, forms / f"{d}.jpg") for d in docs if (forms / f"{d}.jpg").exists()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    done = []
    idx = ["sheet,pos,docket"]
    buf = []
    n = 0
    with ProcessPoolExecutor(a.procs) as ex:
        for d, im in ex.map(one, jobs, chunksize=4):
            if im is None:
                continue
            buf.append((d, im))
            if len(buf) == a.per_sheet:
                n += 1
                tiles = []
                for pos, (dd, t) in enumerate(buf, 1):
                    t = t.copy()
                    t = cv2.copyMakeBorder(t, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    cv2.putText(t, f"#{pos}   TARGET DOCKET ends with ...{dd[-8:]}", (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 0, 0), 2)
                    tiles.append(cv2.copyMakeBorder(t, 0, 10, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)))
                    idx.append(f"{n},{pos},{dd}")
                w = max(t.shape[1] for t in tiles)
                cv2.imwrite(str(out / f"sheet_{n:03d}.jpg"), cv2.cvtColor(np.vstack([cv2.copyMakeBorder(t, 0, 0, 0, w - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]),
                                                                           cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                buf = []
                if n % 20 == 0:
                    (out / "index.csv").write_text("\n".join(idx) + "\n")
                    print("sheets", n, flush=True)
    (out / "index.csv").write_text("\n".join(idx) + "\n")
    print("done: sheets", n, flush=True)


if __name__ == "__main__":
    main()
