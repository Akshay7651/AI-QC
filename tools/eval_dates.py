#!/usr/bin/env python
"""Held-out evaluation of date_reader on never-trained forms (see train_dates.py for the split).
   python tools/eval_dates.py [--crops data/date_crops.pkl] [--gates 0.7,0.8,0.9,0.95]
Truth: hand labels (data/eval/dates.csv: all four fields), app loss date (eval_main, loss only), disputed-case Excel (sowing/loss/intim; eval_disp half)."""
import argparse
import collections
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import date_reader as DR      # noqa: E402
import train_dates as T       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=os.path.join(ROOT, "data", "date_crops.pkl"))
    ap.add_argument("--gates", default="0.0,0.7,0.8,0.9,0.95")
    a = ap.parse_args()
    gates = [float(x) for x in a.gates.split(",")]
    items = [i for i in T.build(a.crops) if i["role"] != "train" and i["date"] is not None]
    t = time.time()
    res = [DR.read_date(i["crop"], i["field"]) for i in items]
    print("%d held-out date boxes, %.1f ms per box" % (len(items), 1000 * (time.time() - t) / len(items)))
    for field in T.FIELDS:
        for role in ("eval_hand", "eval_main", "eval_disp"):
            sel = [(i, r) for i, r in zip(items, res) if i["field"] == field and i["role"] == role]
            if not sel:
                continue
            line = "%-5s %-9s n=%3d exact(all) %4.1f%% |" % (field, role, len(sel), 100 * sum(r["date"] == i["date"].strftime("%d%m%Y") for i, r in sel) / len(sel))
            for g in gates[1:]:
                ans = [(i, r) for i, r in sel if r["date"] and r["conf"] >= g]
                ok = sum(r["date"] == i["date"].strftime("%d%m%Y") for i, r in ans)
                line += " g%.2f cov %3.0f%% prec %5.1f%% (%d/%d) |" % (g, 100 * len(ans) / len(sel), 100 * ok / max(1, len(ans)), ok, len(ans))
            print(line)
    print("\nstyle breakdown (decoded text style, hand+app labels pooled, gate %.2f)" % DR.CONF_GATE)
    st = collections.defaultdict(lambda: [0, 0, 0])
    for i, r in zip(items, res):
        if not r["text"]:
            continue
        ys, short = T.style_of(r["text"])
        for key in ("year " + ys, short):
            st[key][0] += 1
            if r["date"] and r["conf"] >= DR.CONF_GATE:
                st[key][1] += 1
                st[key][2] += r["date"] == i["date"].strftime("%d%m%Y")
    for k, (n, c, ok) in sorted(st.items()):
        print("  %-12s n=%4d answered %4d precision %5.1f%%" % (k, n, c, 100 * ok / max(1, c)))


if __name__ == "__main__":
    main()
