#!/usr/bin/env python3
"""Train the signature-present classifiers (farmer / company / worker) from hand labels + the form reader's ink features.

  python tools/train_sig_classifier.py --cache r_labelled150.pkl r_unseen100.pkl      # pickles of {docket: read_form() result}
Without --cache it runs form_reader.read_form on every labelled form it can find (data/forms, --forms-dir ...), ~2 s/form.
Writes models/sig_clf.joblib (small) and prints 5-fold cross-validated accuracy vs the old ink rule.
Labels: data/eval/sigs.csv and data/eval_unseen100/sigs.csv (1 = pen signature present, 0 = empty; ?/S/NP excluded).
The officer block is NOT learned: it is almost never signed (1 positive in ~246 labelled forms) -> reported as 'not assessed'.
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["area", "area_low", "area_top", "ncomp", "ext_w", "ext_h", "blue", "cy"]
BLOCKS = {"farmer": "farmer_signed", "company": "company_signed", "worker": "worker_signed"}
LABELS = [ROOT / "data/eval/sigs.csv", ROOT / "data/eval_unseen100/sigs.csv"]


def _results(args):
    res = {}
    if args.cache:
        for c in args.cache:
            res.update(pickle.load(open(c, "rb")))
        return res
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    sys.path.insert(0, str(ROOT))
    import form_reader as R
    dirs = [ROOT / "data/forms"] + [Path(d) for d in (args.forms_dir or [])]
    labelled = set()
    for f in LABELS:
        if f.exists():
            labelled |= set(pd.read_csv(f, dtype=str).docket)
    for d in sorted(labelled):
        p = next((x / f"{d}.jpg" for x in dirs if (x / f"{d}.jpg").exists()), None)
        if p:
            res[d] = R.read_form(str(p), docket=d)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="*")
    ap.add_argument("--forms-dir", nargs="*")
    args = ap.parse_args()
    res = _results(args)
    lab = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in LABELS if f.exists()]).drop_duplicates("docket").set_index("docket")
    models, report = {}, {}
    for blk, rule_key in BLOCKS.items():
        X, y, rule = [], [], []
        for d, r in res.items():
            if d not in lab.index or not r.get("_sig_feats") or not r["_sig_feats"].get(blk):
                continue
            if lab.loc[d, blk] not in ("0", "1"):
                continue
            f = r["_sig_feats"][blk]
            X.append([f[k] for k in KEYS]); y.append(int(lab.loc[d, blk])); rule.append(int(bool(r.get(rule_key))))
        X, y, rule = np.array(X), np.array(y), np.array(rule)
        minority = int(min((y == 1).sum(), (y == 0).sum()))
        if minority < 8:
            # too few examples of the rare class to learn from: 'assume the common answer unless the ink is far below
            # anything seen in the common class' (feature 'area' = ink area of the signature box)
            common = int(y.mean() >= 0.5)
            a = X[:, KEYS.index("area")]
            thr = float(0.25 * np.percentile(a[y == common], 5))
            m = {"rule": "common_unless_low_ink", "common": common, "thr": thr, "key": "area"}
            pred = np.where(a < thr, 1 - common, common)
            cv = pred   # no cross-validation possible; this is the in-sample rule accuracy (optimistic)
        else:
            m = GradientBoostingClassifier(n_estimators=150, max_depth=2, random_state=0)
            cv = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
            m.fit(X, y)
        models[blk] = m
        report[blk] = dict(minority_examples=minority, n=len(y), positives=int(y.sum()), cv_acc=round(100 * float((cv == y).mean()), 1),
                           old_rule_acc=round(100 * float((rule == y).mean()), 1),
                           cv_precision_yes=round(100 * float(((cv == 1) & (y == 1)).sum() / max(1, (cv == 1).sum())), 1),
                           cv_recall_yes=round(100 * float(((cv == 1) & (y == 1)).sum() / max(1, (y == 1).sum())), 1))
        print(blk, report[blk])
    out = ROOT / "models" / "sig_clf.joblib"
    joblib.dump({"keys": KEYS, "models": models, "report": report}, out, compress=3)
    print("wrote", out, round(out.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
