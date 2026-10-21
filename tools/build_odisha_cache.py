#!/usr/bin/env python3
"""Build data/digits_cache_odisha/<docket>.npz (raw gray crops of the Affected Area%/Loss% cells) for every
downloaded Odisha form, mirroring data/digits_cache/ for Proforma-3 - same cache shape (ok/area/loss) so
train_cells_odisha.py can reuse train_cells.py's CNN training code almost unchanged.

  python tools/build_odisha_cache.py --forms data/forms_odisha --out data/digits_cache_odisha
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import form_odisha as O
import form_p3 as P


def locate_cells(path):
    """-> (area_gray, loss_gray) raw crops, or (None, None) if the form/table couldn't be located."""
    try:
        rgb = P.load_image(path)
    except Exception:
        return None, None
    r, info = P.normalise(rgb)
    g = cv2.cvtColor(P._resize_w(r, 1600), cv2.COLOR_RGB2GRAY)
    words = O._ocr_data(g)
    if not O.is_odisha_form(words):
        return None, None
    lines = O._group_lines(words)
    box1 = O._find_table_box(lines, O.TABLE1_HDR, next_rx=r"7\)\s*reason\s*of\s*crop\s*loss")
    box2 = O._find_table_box(lines, O.TABLE2_HDR, next_rx=r"16\)\s*loss\s*survey\s*mandatory")
    if box2 is None and box1 is not None:
        box2 = O._find_table_box_by_anchor(lines, r"15\)\s*total\s*information", r"16\)\s*loss\s*survey\s*mandatory",
                                           3, box1["col_bounds"][0], box1["col_bounds"][-1])
    if box2 is None:
        return None, None
    bounds = box2["col_bounds"]
    pad = 4

    def crop(k):
        x0, x1 = bounds[k], bounds[k + 1]
        y0, y1 = box2["row_y0"], box2["row_y1"]
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(g.shape[1], int(x1) + pad), min(g.shape[0], int(y1) + pad)
        if x1 <= x0 or y1 <= y0:
            return None
        return g[y0:y1, x0:x1]
    return crop(0), crop(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default="data/forms_odisha")
    ap.add_argument("--out", default="data/digits_cache_odisha")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.forms, "*.pdf")))
    if a.limit:
        files = files[:a.limit]
    ok = fail = 0
    for i, f in enumerate(files):
        docket = os.path.splitext(os.path.basename(f))[0]
        outp = os.path.join(a.out, docket + ".npz")
        if os.path.exists(outp):
            ok += 1
            continue
        try:
            area, loss = locate_cells(f)
        except Exception:
            area = loss = None
        if area is None or loss is None or area.size < 200 or loss.size < 200:
            np.savez_compressed(outp, ok=0, area=np.zeros((1, 1), np.uint8), loss=np.zeros((1, 1), np.uint8))
            fail += 1
        else:
            np.savez_compressed(outp, ok=1, area=area, loss=loss)
            ok += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(files)}  ok={ok} fail={fail}", flush=True)
    print(f"done: ok={ok} fail={fail} of {len(files)}")


if __name__ == "__main__":
    main()
