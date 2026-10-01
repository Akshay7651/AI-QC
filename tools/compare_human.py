#!/usr/bin/env python3
"""Run the form reader BLIND on human-QC'd forms and compare with the human values (area / loss of the form).

  python tools/compare_human.py --labels Rajasthan_QC_Done.xlsx --forms data/forms_raj --out results/compare_raj.xlsx [--limit N]
Categories per form:  MATCH (AI stated the human values) | WRONG (AI stated other values) | MULTI-MATCH / MULTI-WRONG (several rows on a form:
the AI only checks whether one of the rows equals the human value) | ABSTAIN (AI did not state a value) | NO-LABEL.
Prints precision (of stated values) and coverage, and writes an Excel with the form file path of every row so mismatches can be opened.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_THREAD_LIMIT", "1")


def num(v):
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return None


CACHE = None


def one(args):
    d, path = args
    import pickle
    cf = Path(CACHE) / f"{d}.pkl" if CACHE else None
    if cf is not None and cf.exists():
        try:
            return d, pickle.load(open(cf, "rb"))
        except Exception:     # noqa: BLE001
            pass
    import form_reader as FR
    try:
        r = FR.read_form(path, d)
    except Exception as e:     # noqa: BLE001
        return d, {"err": type(e).__name__}
    keep = {k: r.get(k) for k in ("template", "is_proforma3", "form_area", "form_loss", "row_area", "row_loss", "raj_pairs", "raj_all_read", "confidence",
                                  "farmer_signed", "company_signed", "officer_signed", "quality", "notes")}
    keep["raj_rows"] = r.get("raj_rows")
    keep["field_conf"] = r.get("field_conf")
    if cf is not None:
        cf.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(keep, open(cf, "wb"))
    return d, keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--cache", help="directory that keeps one result per form, so an interrupted run resumes (delete it after changing a model)")
    ap.add_argument("--app-cols", default="affectedAreaPercentage,cropLossPercentage")
    a = ap.parse_args()
    global CACHE
    CACHE = a.cache
    df = pd.read_excel(a.labels, dtype=str) if a.labels.lower().endswith(("xlsx", "xls")) else pd.read_csv(a.labels, dtype=str)
    low = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    dc = low.get("docketid") or low.get("docket")
    hc, hl = low["affectedarea%(form)"], low["croploss%(form)"]
    ac, al = [low.get(x.lower().replace("_", "")) for x in a.app_cols.split(",")]
    forms = Path(a.forms)
    rows = []
    for _, r in df.iterrows():
        d = str(r[dc]).strip()
        p = forms / f"{d}.jpg"
        if p.exists():
            rows.append((d, str(p), num(r[hc]), num(r[hl]), num(r[ac]) if ac else None, num(r[al]) if al else None, r.get("Match/Mismatch (Form&app)"),
                         r.get("Form Status (correct / incomplete/ overwrite)")))
    if a.limit:
        rows = rows[: a.limit]
    print("forms to read:", len(rows), flush=True)
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.procs) as ex:
        res = dict(ex.map(one, [(d, p) for d, p, *_ in rows], chunksize=4))
    out = []
    for d, p, ha, hlv, aa, al_, hm, hs in rows:
        r = res.get(d, {})
        pairs = r.get("raj_pairs")
        ai_a = r.get("form_area") if r.get("form_area") is not None else r.get("row_area")
        ai_l = r.get("form_loss") if r.get("form_loss") is not None else r.get("row_loss")
        if ha is None or hlv is None:
            cat = "NO-LABEL"
        elif ai_a is not None and ai_l is not None:
            cat = "MATCH" if (abs(ai_a - ha) < 1e-6 and abs(ai_l - hlv) < 1e-6) else "WRONG"
        elif pairs:
            hit = any(abs(x - ha) < 1e-6 and abs(y - hlv) < 1e-6 for x, y in pairs)
            cat = "MULTI-MATCH" if hit else ("MULTI-WRONG" if r.get("raj_all_read") else "ABSTAIN")
        else:
            cat = "ABSTAIN"
        out.append({"docket": d, "form_file": p, "category": cat, "human_area": ha, "human_loss": hlv, "app_area": aa, "app_loss": al_, "human_match_text": hm,
                    "human_form_status": hs, "ai_area": ai_a, "ai_loss": ai_l, "ai_pairs": str(pairs) if pairs else "", "template": r.get("template"),
                    "ai_conf": r.get("confidence"), "farmer_signed": r.get("farmer_signed"), "company_signed": r.get("company_signed"),
                    "aao_signed": r.get("officer_signed"), "quality": r.get("quality"), "notes": "; ".join(r.get("notes") or []) if isinstance(r.get("notes"), list) else "",
                    "rows_detail": str([(c["row"], c["area_cls"], c["area_conf"], c["loss_cls"], c["loss_conf"]) for c in (r.get("raj_rows") or []) if not c["empty"]])})
    t = pd.DataFrame(out)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t.to_excel(a.out, index=False)
    lab = t[t.category != "NO-LABEL"]
    stated = lab[lab.category.isin(["MATCH", "WRONG", "MULTI-MATCH", "MULTI-WRONG"])]
    right = stated[stated.category.isin(["MATCH", "MULTI-MATCH"])]
    print(lab.category.value_counts().to_string())
    print(f"\\nlabelled forms {len(lab)} | AI stated {len(stated)} ({100 * len(stated) / max(len(lab), 1):.1f}% coverage) | "
          f"right {len(right)} ({100 * len(right) / max(len(stated), 1):.1f}% precision)")
    z = lab[(lab.human_area > 0) | (lab.human_loss > 0)]
    zs = z[z.category.isin(["MATCH", "WRONG", "MULTI-MATCH", "MULTI-WRONG"])]
    print(f"non-zero forms {len(z)} | stated {len(zs)} | right {len(zs[zs.category.isin(['MATCH', 'MULTI-MATCH'])])}")
    print("written", a.out)


if __name__ == "__main__":
    main()
