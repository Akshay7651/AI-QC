#!/usr/bin/env python
"""Render contact sheets of Proforma-3 crops so a human can hand-label an evaluation set by eye.

Usage (from /home/user/AI-QC):
    python tools/make_label_sheets.py --n 200 --seed 1                # 200 random forms, all four sheet kinds
    python tools/make_label_sheets.py --n 100 --kinds cells,sigs      # only some kinds
    python tools/make_label_sheets.py --dockets 0401...,0401...       # explicit dockets
    python tools/make_label_sheets.py --n 200 --seed 1 --show-app     # also print the app-entered values (biases you! use only for speed-checks)

Outputs (data/eval/sheets/):
    index.csv                      sheet,tile,docket          (tile number -> docket)
    cells_001.png ...              each row = one form: tile#, docket tail, then 6 cells:
                                   area r1 | loss r1 | area r2 | loss r2 | area TOTAL | loss TOTAL
    sigs_001.png ...               each row = one form: the four signature boxes (farmer | company | worker | officer)
                                   with the printed labels underneath
    formno_001.png ...             each row = one form: FORM-NO strip under the barcode + the handwritten PO ID strip
    dates_001.png ...              each row = one form: sowing | loss | intimation | inspection date cells
    templates/{cells,sigs,formno,dates}_todo.csv   empty label files listing every (docket, field) to fill, in sheet order

How to label: open a sheet PNG, read tile number + 8-digit docket tail printed at the left of each row, and fill the
matching template, saving it as data/eval/<name>.csv:
    cells.csv  : docket,field,value     field in {area_r1,loss_r1,area_r2,loss_r2,area_tot,loss_tot}; value = number
                 (as written, e.g. 0, 30, 0.01, 100) or empty when the cell is blank
    sigs.csv   : docket,farmer,company,worker,officer     1 = a real pen signature present, 0 = none,
                 S (officer only) = only a printed rubber stamp, no pen signature
    formno.csv : docket,form_no    e.g. HR0126175631   (also optional po_id.csv: docket,po_id as handwritten, digits only, blank if none)
    dates.csv  : docket,sow,loss,intim,insp   as DDMMYYYY (blank if empty/illegible)
Rows where the layout could not be found are marked LAYOUT FAILED and show the whole page: label them from the page
(data/forms/<docket>.jpg) or write ? in the value.
"""
import argparse
import csv
import glob
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import form_p3 as P  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
ROWS_PER_SHEET = {"cells": 10, "sigs": 8, "formno": 14, "dates": 12}


def _font(sz):
    try:
        return ImageFont.truetype(FONT, sz)
    except Exception:
        return ImageFont.load_default()


def _fit(arr, h, maxw):
    im = Image.fromarray(arr)
    s = h / max(1, im.height)
    w = int(im.width * s)
    if w > maxw:
        s = maxw / im.width
        w = maxw
        h = int(im.height * s)
    return im.resize((max(1, w), max(1, h)), Image.LANCZOS)


def _tiles(docket):
    """-> dict kind -> list of (caption, np RGB) or None per kind when layout failed; uses form_p3 layout"""
    path = os.path.join(ROOT, "data", "forms", docket + ".jpg")
    rgb = P.load_image(path)
    lay = P.analyse_layout(rgb)
    out = {"ok": lay["ok"], "page": None}
    if not lay["ok"]:
        im = Image.fromarray(rgb)
        im.thumbnail((500, 700))
        out["page"] = np.asarray(im)
        return out
    img, B, sc, tab = lay["rgb"], lay["boxes"], lay["scale"], lay["tab"]
    pitch = tab["p"] * sc
    c = lambda k, pad=0: P.crop(img, B[k], pad) if k in B else None  # noqa: E731
    out["cells"] = [("r1 area", c("area_r1", 3)), ("r1 loss", c("loss_r1", 3)), ("r2 area", c("area_r2", 3)),
                    ("r2 loss", c("loss_r2", 3)), ("TOTAL area", c("area_tot", 3)), ("TOTAL loss", c("loss_tot", 3))]
    sig = []
    for nm in ("farmer", "company", "worker", "officer"):
        x0, y0, x1, y1 = B["sig_" + nm]
        _, ly0, _, ly1 = B["lab_" + nm]
        sig.append((nm, img[int(max(0, y0 - 0.6 * pitch)):int(min(img.shape[0], ly1)), int(max(0, x0)):int(min(img.shape[1], x1))]))
    out["sigs"] = sig
    fn = B.get("formno_win")
    out["formno"] = [("FORM NO window", P.crop(img, fn, 0) if fn else None), ("PO ID", c("po_id", 3))]
    out["dates"] = [("sowing", c("sow_date", 3)), ("loss", c("loss_date", 3)), ("intimation", c("intim_date", 3)), ("inspection", c("insp_date", 3))]
    return out


def _render_row(kind, idx, docket, tiles, width, app=None):
    f, fs = _font(26), _font(15)
    lab_w = 210
    if not tiles["ok"]:
        pg = Image.fromarray(tiles["page"])
        row = Image.new("RGB", (width, pg.height + 10), "white")
        row.paste(pg, (lab_w, 5))
        d = ImageDraw.Draw(row)
        d.text((8, 8), f"#{idx}", fill="black", font=f)
        d.text((8, 42), docket[-8:], fill="black", font=fs)
        d.text((8, 70), "LAYOUT FAILED\n(label from page)", fill="red", font=fs)
        return row
    parts = tiles[kind]
    n = len(parts)
    avail = (width - lab_w - 10) // n - 8
    hh = {"cells": 95, "sigs": 230, "formno": 70, "dates": 80}[kind]
    ims = []
    for cap, arr in parts:
        if arr is None or arr.size == 0:
            ims.append((cap, Image.new("RGB", (avail, hh), (230, 230, 230))))
        else:
            ims.append((cap, _fit(arr, hh, avail)))
    if kind == "formno":      # formno window is wide; give it more room
        ims = [(ims[0][0], _fit(parts[0][1], hh, int((width - lab_w) * 0.55))), (ims[1][0], _fit(parts[1][1], hh, int((width - lab_w) * 0.42)))]
    h = max(i.height for _, i in ims) + 22
    row = Image.new("RGB", (width, h), "white")
    d = ImageDraw.Draw(row)
    d.text((8, 4), f"#{idx}", fill="black", font=f)
    d.text((8, 38), docket[-8:], fill=(0, 0, 160), font=fs)
    if app:
        d.text((8, 60), app, fill=(150, 0, 0), font=_font(12))
    x = lab_w
    for cap, im in ims:
        row.paste(im, (x, 18))
        d.text((x, 1), cap, fill=(90, 90, 90), font=fs)
        d.rectangle([x, 18, x + im.width - 1, 18 + im.height - 1], outline=(180, 180, 180))
        x += im.width + 8
    d.line([0, h - 1, width, h - 1], fill=(0, 0, 0), width=2)
    return row


def _work(docket):
    try:
        return docket, _tiles(docket)
    except Exception as e:  # noqa: BLE001
        return docket, {"ok": False, "page": np.full((100, 100, 3), 200, np.uint8), "err": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kinds", default="cells,sigs,formno,dates")
    ap.add_argument("--dockets", default="")
    ap.add_argument("--exclude-labelled", action="store_true", help="skip dockets already in data/eval/*.csv")
    ap.add_argument("--show-app", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval", "sheets"))
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "templates"), exist_ok=True)
    allf = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, "data", "forms", "*.jpg")))
    if a.dockets:
        dk = a.dockets.split(",")
    else:
        done = set()
        if a.exclude_labelled:
            for fn in glob.glob(os.path.join(ROOT, "data", "eval", "*.csv")):
                try:
                    done |= {r["docket"] for r in csv.DictReader(open(fn))}
                except Exception:  # noqa: BLE001
                    pass
        pool = [d for d in allf if d not in done]
        random.Random(a.seed).shuffle(pool)
        dk = sorted(pool[: a.n])
    app = {}
    if a.show_app:
        try:
            for r in csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv"))):
                app[r["docket_id"]] = f"app {r['affected_area_pct']}/{r['crop_loss_pct']}"
        except Exception:  # noqa: BLE001
            pass
    kinds = a.kinds.split(",")
    print(f"{len(dk)} forms, kinds={kinds}", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(_work, dk, chunksize=4))
    W = 1500
    idx_rows = []
    todo = {k: [] for k in kinds}
    for kind in kinds:
        per = ROWS_PER_SHEET[kind]
        for s0 in range(0, len(res), per):
            chunk = res[s0: s0 + per]
            rows = [_render_row(kind, s0 + i + 1, dkt, t, W, app.get(dkt)) for i, (dkt, t) in enumerate(chunk)]
            sh = Image.new("RGB", (W, sum(r.height for r in rows)), "white")
            y = 0
            for r in rows:
                sh.paste(r, (0, y))
                y += r.height
            name = f"{kind}_{s0 // per + 1:03d}.png"
            sh.save(os.path.join(a.out, name))
            for i, (dkt, _) in enumerate(chunk):
                idx_rows.append((name, s0 + i + 1, dkt))
        for dkt, _ in res:
            if kind == "cells":
                todo[kind] += [(dkt, f, "") for f in ("area_r1", "loss_r1", "area_r2", "loss_r2", "area_tot", "loss_tot")]
            elif kind == "sigs":
                todo[kind].append((dkt, "", "", "", ""))
            elif kind == "formno":
                todo[kind].append((dkt, ""))
            else:
                todo[kind].append((dkt, "", "", "", ""))
    hdr = {"cells": ["docket", "field", "value"], "sigs": ["docket", "farmer", "company", "worker", "officer"],
           "formno": ["docket", "form_no"], "dates": ["docket", "sow", "loss", "intim", "insp"]}
    for k, rows in todo.items():
        with open(os.path.join(a.out, "templates", f"{k}_todo.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(hdr[k])
            w.writerows(rows)
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet", "tile", "docket"])
        w.writerows(idx_rows)
    print("wrote sheets to", a.out)
    nfail = sum(1 for _, t in res if not t["ok"])
    print(f"layout failed on {nfail}/{len(res)} forms")


if __name__ == "__main__":
    main()
