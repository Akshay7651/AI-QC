"""Evaluate digits.read_number on the lead's hand-labelled cells (data/eval/cells.csv).

python eval_digits.py [--json out.json] [--show-errors]
Crops come from digits_locate.py (own cell cropper) and are cached in data/digits_cache_eval/.
Reports exact-match overall / non-zero-only / per field, and the pure hold-out = last 50 dockets (never used for tuning).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

import digits as D
import digits_locate as L

HERE = os.path.dirname(os.path.abspath(__file__))
CDIR = os.path.join(HERE, "data", "digits_cache_eval")
FIELDS = {"area_r1": "area", "loss_r1": "loss", "area_r2": "r2_area", "loss_r2": "r2_loss", "area_tot": "t_area", "loss_tot": "t_loss"}


def crops_for(dk):
    os.makedirs(CDIR, exist_ok=True)
    p = os.path.join(CDIR, dk + ".npz")
    if not os.path.exists(p):
        img = Image.open(os.path.join(HERE, "data", "forms", dk + ".jpg"))
        cc = L.cell_crops(img)
        if cc is None:
            np.savez_compressed(p, ok=np.array(0))
        else:
            np.savez_compressed(p, ok=np.array(1), **{k: np.asarray(v) for k, v in cc.items() if hasattr(v, "size") or isinstance(v, Image.Image)})
    z = np.load(p)
    return None if not int(z["ok"]) else {k: z[k] for k in z.files if k != "ok"}


def run(show=False, reader=None):
    reader = reader or D.read_number
    lab = pd.read_csv(os.path.join(HERE, "data", "eval", "cells.csv"), dtype=str, keep_default_na=False)
    lab = lab[lab.field.isin(FIELDS)]
    dockets = sorted(set(lab.docket))
    hold = set(dockets[-50:])
    rows = []
    for dk in dockets:
        if not os.path.exists(os.path.join(HERE, "data", "forms", dk + ".jpg")):
            continue
        cc = crops_for(dk)
        for _, r in lab[lab.docket == dk].iterrows():
            v = r["value"].strip()
            if v == "?":
                continue
            exp = None if v == "" else float(v)
            if cc is None or FIELDS[r["field"]] not in cc:
                out = dict(value=None, text="", conf=0.0, n_components=0)
            else:
                out = reader(Image.fromarray(cc[FIELDS[r["field"]]]))
            ok = (exp is None and out["value"] is None) or (exp is not None and out["value"] is not None and abs(out["value"] - exp) < 1e-6)
            rows.append(dict(docket=dk, field=r["field"], exp=exp, got=out["value"], text=out["text"], conf=out["conf"], ok=ok, hold=dk in hold))
    df = pd.DataFrame(rows)
    # forms with any non-zero value
    nz = set(df[(df.exp.notna()) & (df.exp != 0)].docket)
    df["nz_form"] = df.docket.isin(nz)
    res = {}
    def acc(d):
        return dict(n=int(len(d)), acc=round(float(d.ok.mean()), 4) if len(d) else None)
    main = df[df.field.isin(["area_r1", "loss_r1", "area_tot", "loss_tot"])]
    res["all_cells"] = acc(df)
    res["filled_cells(exp not blank)"] = acc(df[df.exp.notna()])
    res["blank_cells"] = acc(df[df.exp.isna()])
    res["nonzero_value_cells"] = acc(df[(df.exp.notna()) & (df.exp != 0)])
    res["zero_value_cells"] = acc(df[df.exp == 0])
    res["row1_cells(area+loss)"] = acc(df[df.field.isin(["area_r1", "loss_r1"])])
    res["row1_nonzero_forms"] = acc(df[df.field.isin(["area_r1", "loss_r1"]) & df.nz_form])
    res["row1_filled_only"] = acc(df[df.field.isin(["area_r1", "loss_r1"]) & df.exp.notna()])
    res["HOLDOUT_last50_all"] = acc(df[df.hold])
    res["HOLDOUT_last50_filled"] = acc(df[df.hold & df.exp.notna()])
    res["HOLDOUT_last50_nonzero_value"] = acc(df[df.hold & df.exp.notna() & (df.exp != 0)])
    # form-level rule: value = total row if filled else row 1 (area and loss separately)
    gate_rows = []
    for fld in ("area", "loss"):
        for dk, g in df.groupby("docket"):
            t = g[g.field == fld + "_tot"]; r1 = g[g.field == fld + "_r1"]
            if r1.empty or t.empty:
                continue
            exp_t, exp_r = t.iloc[0].exp, r1.iloc[0].exp
            truth = exp_t if not pd.isna(exp_t) else exp_r
            if pd.isna(truth):
                continue
            got_t, got_r = t.iloc[0].got, r1.iloc[0].got
            got = got_t if not pd.isna(got_t) else got_r
            gate_rows.append(dict(docket=dk, field=fld, truth=truth, got=got, conf=min(t.iloc[0].conf, r1.iloc[0].conf), hold=dk in hold))
    g = pd.DataFrame(gate_rows)
    if len(g):
        g["ok"] = g.got.notna() & ((g.got - g.truth).abs() < 1e-6)
        res["FORM_LEVEL(total else row1)"] = dict(n=len(g), exact=round(float(g.ok.mean()), 4),
            nonzero_n=int((g.truth != 0).sum()), nonzero_exact=round(float(g[g.truth != 0].ok.mean()), 4),
            zero_exact=round(float(g[g.truth == 0].ok.mean()), 4), answered=round(float(g.got.notna().mean()), 4),
            precision_when_answered=round(float(g[g.got.notna()].ok.mean()), 4) if g.got.notna().any() else None,
            holdout_exact=round(float(g[g.hold].ok.mean()), 4), holdout_n=int(g.hold.sum()))
    res["per_field"] = {f: acc(df[df.field == f]) for f in FIELDS}
    # value-level: 'wrong' vs 'None' (abstained) among errors
    err = df[~df.ok]
    res["errors"] = dict(n=int(len(err)), abstained=int(err.got.isna().sum() & 0) + int((err.got.isna() & err.exp.notna()).sum()), wrong_value=int((err.got.notna()).sum()))
    if show:
        print(err[["docket", "field", "exp", "got", "text", "conf"]].to_string())
    return res, df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--json"); ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--cell", action="store_true", help="use the whole-cell classifier")
    a = ap.parse_args()
    res, df = run(a.show_errors, D.read_cell if a.cell else None)
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
