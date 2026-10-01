#!/usr/bin/env python
"""Extract the 4 date-box crops (grey, kept at native res up to 160 px high) for every form in data/forms and data/disputed/forms.
   python tools/extract_date_crops.py [--out data/date_crops.npz]"""
import argparse, os, sys, pickle
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KEYS = ["sow_date", "loss_date", "intim_date", "insp_date"]

def work(path):
    import form_p3 as P
    try:
        rgb = P.load_image(path)
        lay = P.analyse_layout(rgb)
        if not lay["ok"]:
            return path, None
        B, r = lay["boxes"], lay["rgb"]
        out = []
        for k in KEYS:
            if k not in B:
                out.append(None); continue
            c = P.crop(r, B[k], 2)
            g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
            if g.shape[0] > 160:
                s = 160 / g.shape[0]; g = cv2.resize(g, (max(8, int(g.shape[1]*s)), 160), interpolation=cv2.INTER_AREA)
            out.append(g)
        return path, out
    except Exception:
        return path, None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "data", "date_crops.pkl"))
    ap.add_argument("--skip", default=None, help="pickle of an earlier run: its forms are not extracted again")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    paths = []
    for d, src in (("forms", "main"), ("disputed/forms", "disp")):
        dd = os.path.join(ROOT, "data", d)
        paths += [(os.path.join(dd, f), src) for f in sorted(os.listdir(dd)) if f.endswith(".jpg")]
    import csv, random
    hand = {r["docket"].zfill(18) for r in csv.DictReader(open(os.path.join(ROOT, "data", "eval", "dates.csv")))}
    random.Random(0).shuffle(paths)          # priority: disputed + hand-labelled first, then the rest in random order
    paths.sort(key=lambda t: 0 if (t[1] == "disp" or os.path.basename(t[0])[:-4] in hand) else 1)
    if a.skip:
        done = pickle.load(open(a.skip, "rb"))
        paths = [t for t in paths if (t[1], os.path.basename(t[0])[:-4]) not in done]
    res = {}
    with ProcessPoolExecutor(a.procs) as ex:
        for i, (p, o) in enumerate(ex.map(work, [p for p, _ in paths], chunksize=4)):
            src = paths[i][1]
            res[(src, os.path.basename(p)[:-4])] = o
            if i % 250 == 249:
                print(i, len(paths), flush=True)
                pickle.dump(res, open(a.out + ".tmp", "wb")); os.replace(a.out + ".tmp", a.out)
    pickle.dump(res, open(a.out, "wb"))
    print("done", sum(v is not None for v in res.values()), len(res))
