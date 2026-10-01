#!/usr/bin/env python3
"""Compare an AI-QC output Excel with hand-labelled ground truth and print per-field precision / coverage.

  python tools/accuracy_report.py --output output/unseen100_QC.xlsx --labels data/eval_unseen100 [--json report.json]

Label files (any may be missing): cells.csv (docket,field,value), sigs.csv (docket,farmer,company,worker,officer),
formno.csv (docket,form_no), photos.csv (docket,photo_is_form,scene,crop_visible).
Definitions: coverage = share of labelled items the system dared to answer; precision = share of ANSWERED items that are right;
accuracy_incl_abstain = correct / all labelled (an abstention counts as wrong). '?' labels are excluded.
Target (user): precision >= 95% on every asserted field.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def summarize(name, n, answered, correct, extra=None):
    d = {"field": name, "labelled": n, "answered": answered, "coverage_%": pct(answered, n),
         "precision_%": pct(correct, answered), "accuracy_incl_abstain_%": pct(correct, n)}
    if extra:
        d.update(extra)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--json")
    a = ap.parse_args()
    o = pd.read_excel(a.output, dtype=str)
    o["docket_id"] = o["docket_id"].astype(str).str.strip()
    o = o.set_index("docket_id")
    L = Path(a.labels)
    rows = []

    def col(*names):
        for n in names:
            for c in o.columns:
                if c.lower().startswith(n.lower()):
                    return c
        return None

    # ---- cells (area / loss): label rule = total row if filled else row 1
    f = L / "cells.csv"
    if f.exists():
        c = pd.read_csv(f, dtype=str, keep_default_na=False)
        np_forms = set(c.loc[c.field == "NOT_PROFORMA3", "docket"])
        p = c.pivot(index="docket", columns="field", values="value")
        ca, cl = col("Affected area% (Form)"), col("Crop Loss% (Form)")
        for nm, tot, r1, oc in (("area %", "area_tot", "area_r1", ca), ("loss %", "loss_tot", "loss_r1", cl)):
            n = ans = ok = nz = nzans = nzok = 0
            for d, r in p.iterrows():
                if d in np_forms or d not in o.index:
                    continue
                lab = num(r.get(tot, ""))
                if np.isnan(lab):
                    lab = num(r.get(r1, ""))
                if np.isnan(lab):
                    continue
                n += 1
                v = num(o.loc[d, oc]) if oc else np.nan
                if lab > 0:
                    nz += 1
                if not np.isnan(v):
                    ans += 1
                    ok += int(abs(v - lab) < 1e-6)
                    if lab > 0:
                        nzans += 1
                        nzok += int(abs(v - lab) < 1e-6)
            rows.append(summarize(nm, n, ans, ok, {"non_zero_labelled": nz, "non_zero_answered": nzans, "non_zero_correct": nzok}))

    # ---- form number
    f = L / "formno.csv"
    if f.exists():
        c = pd.read_csv(f, dtype=str, keep_default_na=False)
        c = c[~c.form_no.isin(["?", "NONE", ""])]
        oc = col("Form No")
        n = ans = ok = 0
        for _, r in c.iterrows():
            if r.docket not in o.index:
                continue
            n += 1
            v = o.loc[r.docket, oc] if oc else None
            if isinstance(v, str) and re.match(r"HR\d{10}$", v.strip()):
                ans += 1
                ok += int(v.strip() == r.form_no.strip())
        rows.append(summarize("form no", n, ans, ok))

    # ---- signatures
    f = L / "sigs.csv"
    if f.exists():
        c = pd.read_csv(f, dtype=str, keep_default_na=False)
        cols = {"farmer": col("Farmer Signature"), "company": col("Surveyor Signature"),
                "worker": col("Primary Worker Signature"), "officer": col("Government Signature", "Government (Block")}
        for k, oc in cols.items():
            n = ans = ok = pos = tp = fp = 0
            for _, r in c.iterrows():
                lab = r[k]
                if lab not in ("0", "1") or r.docket not in o.index or oc is None:
                    continue
                n += 1
                pos += lab == "1"
                v = str(o.loc[r.docket, oc]).strip().lower()
                if v in ("yes", "no"):
                    ans += 1
                    ok += int((v == "yes") == (lab == "1"))
                    tp += int(v == "yes" and lab == "1")
                    fp += int(v == "yes" and lab == "0")
            rows.append(summarize(f"signature {k}", n, ans, ok, {"labelled_present": pos, "precision_of_yes_%": pct(tp, tp + fp)}))

    # ---- photos
    f = L / "photos.csv"
    if f.exists():
        c = pd.read_csv(f, dtype=str, keep_default_na=False)
        oc = col("Photo is form image")
        n = ans = ok = 0
        for _, r in c.iterrows():
            if r.photo_is_form not in ("all", "some", "none") or r.docket not in o.index or oc is None:
                continue
            n += 1
            v = str(o.loc[r.docket, oc]).strip().lower()
            if v in ("yes", "no"):
                ans += 1
                ok += int((v == "yes") == (r.photo_is_form == "all"))   # system flags 'yes' only when ALL photos are the form
        rows.append(summarize("photo is the paper form (all photos)", n, ans, ok))

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.to_string(index=False))
    if "QC Verdict" in o.columns:
        print("\nVerdicts:", o["QC Verdict"].value_counts().to_dict())
    low = t[(t["precision_%"].notna()) & (t["precision_%"] < 95)]
    print("\nFields BELOW 95% precision:", ", ".join(low.field) if len(low) else "none")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
