#!/usr/bin/env python
"""Evaluate form_reader.read_form against the hand labels in data/eval (cells.csv sigs.csv formno.csv dates.csv).

    python tools/eval_form_reader.py                # run the reader on every labelled docket (3 processes) and score
    python tools/eval_form_reader.py --cache r.pkl  # reuse / write a pickle of reader outputs
    python tools/eval_form_reader.py --heldout 50   # the last N labelled dockets (sorted) are reported separately
                                                    # and must never be used for tuning

Rules: 'form value' for area/loss = TOTAL row if filled else row 1 (first filled row). '?' / NP / NONE / blank labels
are excluded from accuracy (NP/NONE rows are used for is_proforma3 / form_no-absent checks).
"""
import argparse
import csv
import os
import pickle
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EV = os.path.join(ROOT, "data", "eval")


def _rd(name):
    p = os.path.join(EV, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def _work(d):
    import form_reader as R
    t = time.process_time()
    w = time.time()
    try:
        o = R.read_form(os.path.join(ROOT, "data", "forms", d + ".jpg"), d)
    except Exception as e:     # noqa: BLE001
        o = {"error": repr(e)}
    o["cpu"] = time.process_time() - t
    o["wall"] = time.time() - w
    return d, o


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def acc(pairs):
    n = len(pairs)
    return (sum(1 for a, b in pairs if a == b) / n if n else float("nan")), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--heldout", type=int, default=50)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    sigs = {r["docket"]: r for r in _rd("sigs.csv")}
    formno = {r["docket"]: r["form_no"] for r in _rd("formno.csv")}
    dates = {r["docket"]: r for r in _rd("dates.csv")}
    cells = defaultdict(dict)
    for r in _rd("cells.csv"):
        cells[r["docket"]][r["field"]] = r["value"]
    dockets = sorted(set(sigs) | set(formno) | set(dates) | set(cells))
    held = set(dockets[-a.heldout:]) if a.heldout else set()
    if a.cache and os.path.exists(a.cache):
        res = pickle.load(open(a.cache, "rb"))
    else:
        with ProcessPoolExecutor(a.jobs) as ex:
            res = dict(ex.map(_work, dockets))
        if a.cache:
            pickle.dump(res, open(a.cache, "wb"))
    for name, sel in (("ALL", dockets), ("TUNE (non held-out)", [d for d in dockets if d not in held]), ("HELD-OUT", [d for d in dockets if d in held])):
        if not sel:
            continue
        print(f"\n===== {name}: {len(sel)} forms =====")
        ok = [res[d] for d in sel if "error" not in res[d]]
        print(f"reader errors: {len(sel) - len(ok)}; cpu/form mean {sum(o['cpu'] for o in ok) / max(1, len(ok)):.2f}s  median "
              f"{sorted(o['cpu'] for o in ok)[len(ok) // 2]:.2f}s")
        # is_proforma3
        pp = [(formno.get(d) not in ("NONE",) and sigs.get(d, {}).get("farmer") != "NP", bool(res[d].get("is_proforma3"))) for d in sel if d in res and "error" not in res[d]]
        print("is_proforma3 agreement (label: not NP):", acc(pp))
        # form no
        pairs = [(formno[d], res[d].get("form_no")) for d in sel if formno.get(d) and formno[d] not in ("?", "NONE") and "error" not in res[d]]
        print("form_no exact:", acc(pairs), " reader returned None on", sum(1 for a_, b_ in pairs if b_ is None))
        hi = [(x, y) for (x, y), d in zip(pairs, [d for d in sel if formno.get(d) and formno[d] not in ("?", "NONE") and "error" not in res[d]]) if res[d].get("form_no_conf", 0) >= 0.85]
        print("form_no exact among conf>=0.85:", acc(hi))
        # signatures
        for fld, key in (("farmer", "farmer_signed"), ("company", "company_signed"), ("worker", "worker_signed"), ("officer", "officer_signed")):
            pr = []
            for d in sel:
                if d in sigs and sigs[d][fld] in ("0", "1", "S") and "error" not in res[d] and res[d].get("is_proforma3"):
                    lab = sigs[d][fld] == "1"
                    pr.append((lab, bool(res[d].get(key))))
            tp = sum(1 for l, p in pr if l and p)
            fp = sum(1 for l, p in pr if not l and p)
            fn = sum(1 for l, p in pr if l and not p)
            print(f"sig {fld:8s} acc {acc(pr)[0]:.3f} n={len(pr)}  present: label={tp + fn} prec={tp / max(1, tp + fp):.2f} rec={tp / max(1, tp + fn):.2f}")
        st = [(sigs[d]["officer"] == "S", bool(res[d].get("officer_stamp_only"))) for d in sel if d in sigs and sigs[d]["officer"] in ("0", "1", "S") and "error" not in res[d]]
        print("officer stamp-only acc:", acc(st), " labelled S:", sum(1 for l, _ in st if l))
        # cells: form value rule = total row if filled else first filled row
        nz, allv, blank = [], [], []
        for d in sel:
            c = cells.get(d)
            if not c or "error" in res[d] or not res[d].get("is_proforma3"):
                continue
            ok_lab = all(c.get(k) != "?" for k in ("area_tot", "loss_tot", "area_r1", "loss_r1"))
            if not ok_lab:
                continue
            def lv(k):
                return _num(c.get(k))
            la, ll = (lv("area_tot"), lv("loss_tot")) if (lv("area_tot") is not None or lv("loss_tot") is not None) else (lv("area_r1"), lv("loss_r1"))
            tot_blank_lab = lv("area_tot") is None and lv("loss_tot") is None
            blank.append((tot_blank_lab, bool(res[d].get("total_row_blank"))))
            r_ = res[d]
            pa, pl = (r_.get("form_area"), r_.get("form_loss")) if not r_.get("total_row_blank") else (r_.get("row_area"), r_.get("row_loss"))
            both = (la == pa) and (ll == pl)
            allv.append(both)
            if (la or 0) > 0 or (ll or 0) > 0:
                nz.append(both)
        print(f"area/loss pair exact (rule: total row else row1): overall {sum(allv) / max(1, len(allv)):.3f} n={len(allv)} | non-zero forms {sum(nz) / max(1, len(nz)):.3f} n={len(nz)}")
        print("total_row_blank acc:", acc(blank))
        # dates
        for lab, key in (("sow", "sow_date"), ("loss", "loss_date"), ("intim", "intimation_date"), ("insp", "inspection_date")):
            pr = [(dates[d][lab], res[d].get(key) or "") for d in sel if d in dates and dates[d][lab] not in ("?",) and "error" not in res[d] and res[d].get("is_proforma3")
                  and not any(v == "?" for v in dates[d].values())]
            print(f"date {lab:6s}", acc(pr))
        # anchor detection
        print("layout/is_proforma3 found on labelled forms (label not NP):", acc([(True, bool(res[d].get("is_proforma3"))) for d in sel if d in res and sigs.get(d, {}).get("farmer") != "NP"]))


if __name__ == "__main__":
    main()
