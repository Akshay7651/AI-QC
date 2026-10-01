#!/usr/bin/env python3
"""Contact sheets of the signature region of Rajasthan forms, for hand labelling (farmer | company TC/DC | AAO).
  python tools/make_sig_sheets_raj.py --n 400 --per-sheet 6 --out data/eval_raj/sheets
Writes sheet_001.jpg ... and index.csv (sheet, pos, docket).  Labels go to data/eval_raj/sigs.csv  (docket,farmer,company,aao ; 1/0/?).
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def strip(path):
    import form_p3 as P
    import form_raj as R
    lay = P.analyse_layout(P.load_image(str(path)))
    if not lay["ok"]:
        return None
    geo = R.rows_geometry(lay)
    ce = R.column_edges(lay, *geo) if geo else None
    if not geo or not ce:
        return None
    y0, p = geo
    sc = lay["scale"]
    L, Rr = ce[0][0], ce[0][-1]
    ya, yb = (y0 + 10 * p + 3.0 * p) * sc, (y0 + 10 * p + 10.0 * p) * sc
    r = lay["rgb"]
    s = r[int(max(0, ya)): int(min(r.shape[0], yb)), int(max(0, L * sc - 10)): int(min(r.shape[1], Rr * sc + 10))]
    if s.size == 0:
        return None
    return cv2.resize(s, (1000, max(60, int(s.shape[0] * 1000 / s.shape[1]))))


def one(args):
    d, f = args
    try:
        return d, strip(f)
    except Exception:     # noqa: BLE001
        return d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default=str(ROOT / "data/forms_raj"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--per-sheet", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "data/eval_raj/sheets"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    files = sorted(Path(a.forms).glob("*.jpg"))
    random.Random(a.seed).shuffle(files)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.procs) as ex:
        res = [x for x in ex.map(one, [(f.stem, f) for f in files[: int(a.n * 1.3)]], chunksize=4) if x[1] is not None][: a.n]
    idx = ["sheet,pos,docket"]
    for si in range(0, len(res), a.per_sheet):
        chunk = res[si: si + a.per_sheet]
        tiles = []
        for pos, (d, im) in enumerate(chunk, 1):
            t = im.copy()
            cv2.rectangle(t, (0, 0), (1000, 34), (255, 255, 255), -1)
            cv2.putText(t, f"#{pos}  ...{d[-8:]}", (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 0, 0), 2)
            for k in (1, 2):                       # thirds
                cv2.line(t, (k * 333, 34), (k * 333, t.shape[0]), (0, 160, 0), 2)
            tiles.append(cv2.copyMakeBorder(t, 0, 8, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)))
            idx.append(f"{si // a.per_sheet + 1},{pos},{d}")
        cv2.imwrite(str(out / f"sheet_{si // a.per_sheet + 1:03d}.jpg"), cv2.cvtColor(np.vstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (out / "index.csv").write_text("\n".join(idx) + "\n")
    print("sheets:", (len(res) + a.per_sheet - 1) // a.per_sheet, "forms:", len(res))


if __name__ == "__main__":
    main()
