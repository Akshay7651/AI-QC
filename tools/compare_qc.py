#!/usr/bin/env python3
"""Compare two QC outputs of run_qc.py on the same input, column by column, docket by docket.

  python tools/compare_qc.py --input Input/2000.xlsx --ref Input/2000_ClaudeQC.xlsx --test Input/2000_AIQC.xlsx --out Input/compare_2000.xlsx

--ref  = the answers taken as correct (Claude QC, or a human QC sheet with the same column names)
--test = the run being judged (AI-QC, --engine local)
Every docket of --input appears in the result (nothing skipped): dockets missing from --ref or --test are listed as such.
Per QC column and docket the status is one of
  AGREE | DISAGREE | TEST_EMPTY (AI gave no answer) | REF_EMPTY (reference gave no answer) | BOTH_EMPTY | MISSING_ROW
Free-text columns (remarks) are only compared on filled / empty.
Output Excel: sheet "summary" (one line per column), sheet "detail" (one line per docket, ref | test | status for each column),
sheet "disagree" (only dockets with at least one DISAGREE), plus <out>_disagree_dockets.txt for collecting those forms.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

COLS = [  # (column, kind)
    ("Affected area% (Form)", "num"),
    ("Crop Loss% (Form)", "num"),
    ("Match/Mismatch (Form&app)", "cat"),
    ("Date of survey (as per Geo Tagged Image)", "date"),
    ("Field photo (no crop / cut & spread / crop mismatch / standing crop)", "cat"),
    ("Surveyor Signature (Yes/No)", "yn"),
    ("Farmer Signature (Yes/No)", "yn"),
    ("Government Signature (Yes/No)", "yn"),
    ("Form Status (correct / incomplete/ overwrite)", "cat"),
    ("Farmer Photo (Yes/No)", "yn"),
    ("Loss as per Photo (Yes/No)", "yn"),
    ("PO ID matches docket", "yn"),
    ("Survey remarks on form", "text"),
    ("Any Other Remarks", "text"),
]


def docket(v):
    s = str(v).strip().lstrip("'").strip()
    if s.endswith(".0"):
        s = s[:-2]
    return "" if s.lower() == "nan" else s


def empty(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip().lower() in ("", "nan", "none", "na", "n/a", "-")


def norm(v, kind):
    if empty(v):
        return None
    s = str(v).strip().lower()
    if kind == "num":
        m = re.search(r"\d+(\.\d+)?", s.replace(",", "."))
        return float(m.group()) if m else s
    if kind == "yn":
        if s.startswith(("y", "true", "1", "present")):
            return "yes"
        if s.startswith(("n", "false", "0", "absent")):
            return "no"
        return s
    if kind == "date":
        iso = re.match(r"\d{4}-\d{2}-\d{2}", s)
        d = pd.to_datetime(s[:10], format="%Y-%m-%d", errors="coerce") if iso else pd.to_datetime(s, dayfirst=True, errors="coerce")
        return d.strftime("%Y-%m-%d") if not pd.isna(d) else s
    if kind == "text":
        return "filled"
    return re.sub(r"\s+", " ", s)


def load(path, need_docket=True):
    df = pd.read_excel(path, dtype=str) if str(path).lower().endswith((".xlsx", ".xlsm", ".xls")) else pd.read_csv(path, dtype=str)
    low = {str(c).strip().lower(): c for c in df.columns}
    dc = low.get("docket_id") or low.get("docketid") or low.get("docket")
    if dc is None and need_docket:
        sys.exit(f"{path}: no Docket_ID column. Columns: {', '.join(map(str, df.columns[:30]))}")
    df["_d"] = df[dc].map(docket)
    dup = df["_d"].duplicated() & df["_d"].ne("")
    if dup.any():
        print(f"WARNING {Path(path).name}: {int(dup.sum())} duplicate docket rows, the first one is used", flush=True)
    return df[~dup].set_index("_d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="the original Excel (defines the full docket list)")
    ap.add_argument("--ref", required=True, help="reference QC output (e.g. Claude QC)")
    ap.add_argument("--test", required=True, help="QC output being judged (e.g. AI-QC local)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inp, ref, tst = load(a.input), load(a.ref), load(a.test)
    dockets = [d for d in inp.index if d]
    cols = [(c, k) for c, k in COLS if c in ref.columns or c in tst.columns]
    print(f"dockets in input {len(dockets)} | in ref {sum(d in ref.index for d in dockets)} | in test {sum(d in tst.index for d in dockets)}")
    missing_cols = [c for c, _ in COLS if c not in ref.columns and c not in tst.columns]
    if missing_cols:
        print("columns in neither output (not compared):", "; ".join(missing_cols))

    rows = []
    for d in dockets:
        r = {"Docket_ID": d}
        for c, k in cols:
            rv = ref.at[d, c] if (d in ref.index and c in ref.columns) else None
            tv = tst.at[d, c] if (d in tst.index and c in tst.columns) else None
            if d not in ref.index or d not in tst.index:
                st = "MISSING_ROW"
            else:
                rn, tn = norm(rv, k), norm(tv, k)
                st = ("BOTH_EMPTY" if rn is None and tn is None else "TEST_EMPTY" if tn is None else "REF_EMPTY" if rn is None
                      else "AGREE" if rn == tn else "DISAGREE")
            r[f"{c} | ref"] = rv
            r[f"{c} | test"] = tv
            r[f"{c} | status"] = st
        rows.append(r)
    det = pd.DataFrame(rows)

    summ = []
    for c, k in cols:
        vc = det[f"{c} | status"].value_counts()
        both = int(vc.get("AGREE", 0) + vc.get("DISAGREE", 0))
        summ.append({"column": c, "kind": k, **{s: int(vc.get(s, 0)) for s in ("AGREE", "DISAGREE", "TEST_EMPTY", "REF_EMPTY", "BOTH_EMPTY", "MISSING_ROW")},
                     "test answered % (of ref answered)": round(100 * both / max(1, both + int(vc.get("TEST_EMPTY", 0))), 1),
                     "agreement % (where both answered)": round(100 * int(vc.get("AGREE", 0)) / max(1, both), 1)})
    summ = pd.DataFrame(summ)
    stcols = [f"{c} | status" for c, _ in cols]
    bad = det[det[stcols].eq("DISAGREE").any(axis=1)]
    out = Path(a.out)
    with pd.ExcelWriter(out) as w:
        summ.to_excel(w, sheet_name="summary", index=False)
        det.to_excel(w, sheet_name="detail", index=False)
        bad.to_excel(w, sheet_name="disagree", index=False)
    (out.with_name(out.stem + "_disagree_dockets.txt")).write_text("\n".join(bad["Docket_ID"]) + "\n")
    pd.set_option("display.width", 250)
    print(summ.to_string(index=False))
    print(f"\nrows in report {len(det)} (= dockets in input) | dockets with at least one DISAGREE: {len(bad)}")
    print(f"written {out} and {out.stem}_disagree_dockets.txt")


if __name__ == "__main__":
    main()
