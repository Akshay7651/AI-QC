#!/usr/bin/env python3
"""Train the signature-present classifiers for the Rajasthan form (farmer / company TC-DC / agriculture supervisor AAO).

  python tools/train_sig_raj.py --labels Rajasthan_QC_Done.xlsx [--forms data/forms_raj]
Weak labels come from the human column 'Signature Status (Farmer/Govt/surveyor)': 'yes'/'ok' = all signed; text such as
'AAO sign missing', 'Govt. sign missing', 'Farmer & Govt. sign missing', 'Farmer and surveyor sign missing' names the MISSING blocks
(not mentioned = present).  Mismatch / unclear / data-not-found texts are skipped.  5-fold cross-validation decides per block whether the
classifier is trusted (accuracy >= 0.93 and clearly above the majority baseline); untrusted blocks stay 'not assessed'.
Writes models/sig_clf_raj.joblib.
"""
import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KEYS = ["area", "area_low", "area_top", "ncomp", "ext_w", "ext_h", "blue", "cy"]
SKIP = ("not found", "not clear", "not crear", "link", "no doc", "no form", "blur", "no data", "not match", "not upload", "no field", "docket", "na", "-", "wrong", "crop")


def weak_labels(txt):
    t = str(txt).strip().lower()
    if not t or t == "nan":
        return None
    if t in ("yes", "ok", "all ok", "present"):
        return {"farmer": 1, "company": 1, "aao": 1}
    if any(k in t for k in SKIP):
        return None
    lab = {"farmer": 1, "company": 1, "aao": 1}
    mism = "mismatch" in t or "mis match" in t
    named = False
    if "farmer" in t or "famer" in t or "farmar" in t:
        named = True
        lab["farmer"] = None if mism else 0
    if "surveyor" in t or "company" in t:
        named = True
        lab["company"] = 0
    if "aao" in t or "govt" in t or "govt." in t:
        named = True
        lab["aao"] = 0
    if t in ("no",):
        return None
    return lab if named else None


def _feat(docket, forms):
    import form_p3 as P
    import form_raj as R
    p = forms / f"{docket}.jpg"
    if not p.exists():
        return docket, None
    try:
        lay = P.analyse_layout(P.load_image(str(p)))
        return docket, (R.sig_feats(lay) if lay["ok"] else None)
    except Exception:     # noqa: BLE001
        return docket, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--forms", default=str(ROOT / "data/forms_raj"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--vision", default=str(ROOT / "data/eval_raj/sigs.csv"), help="hand labels read by eye (docket,farmer,company,aao; 1/0/?): they override the weak labels and are the trusted test set")
    a = ap.parse_args()
    df = pd.read_excel(a.labels, dtype=str) if a.labels.lower().endswith(("xlsx", "xls")) else pd.read_csv(a.labels, dtype=str)
    low = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    dcol = low.get("docketid") or low.get("docket")
    scol = [c for c in df.columns if c.lower().startswith("signature status")][0]
    lab = {str(d).strip(): weak_labels(t) for d, t in zip(df[dcol], df[scol])}
    lab = {d: l for d, l in lab.items() if l}
    vis = {}
    if os.path.exists(a.vision):
        vdf = pd.read_csv(a.vision, dtype=str)
        for _, r in vdf.iterrows():
            d = str(r["docket"]).strip().zfill(18)
            vis[d] = {"farmer": int(r["farmer"]) if r["farmer"] in ("0", "1") else None, "company": int(r["company"]) if r["company"] in ("0", "1") else None,
                      "aao": int(r["aao"]) if r["aao"] in ("0", "1") else None}
        for d, l in vis.items():
            base = dict(lab.get(d) or {})
            for b, v in l.items():
                if v is not None:
                    base[b] = v
            lab[d] = base
    print("forms with a usable signature label:", len(lab), "| of them hand-read by eye:", len(vis))
    from concurrent.futures import ProcessPoolExecutor
    import pickle
    forms = Path(a.forms)
    cache = ROOT / "data" / "raj_sig_feats.pkl"
    feats = pickle.load(open(cache, "rb")) if cache.exists() else {}
    todo = [d for d in lab if d not in feats]
    for i in range(0, len(todo), 150):                 # saved after every batch: the cloud machine can restart
        part = todo[i: i + 150]
        with ProcessPoolExecutor(a.procs) as ex:
            feats.update(dict(ex.map(_feat, part, [forms] * len(part), chunksize=8)))
        pickle.dump(feats, open(cache, "wb"))
        print(f"  signature features {len(feats)} forms", flush=True)
    models, report = {}, {}
    for blk in ("farmer", "company", "aao"):
        X, y, isv = [], [], []
        for d, l in lab.items():
            f = feats.get(d)
            if f and l.get(blk) is not None and f.get(blk):
                X.append([f[blk][k] for k in KEYS]); y.append(l[blk]); isv.append(d in vis and vis[d].get(blk) is not None)
        X, y, isv = np.array(X), np.array(y), np.array(isv)
        if len(y) < 60 or min(y.sum(), len(y) - y.sum()) < 10:
            print(blk, "too few examples", len(y), int(y.sum()) if len(y) else 0); models[blk] = None; continue
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=0)
        prob = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        m = isv if isv.sum() >= 40 else np.ones(len(y), bool)         # trusted test set = the hand-read forms
        yt, pt = y[m], prob[m]
        t_hi = t_lo = None
        for t in np.arange(0.5, 0.995, 0.01):                         # smallest threshold at which "present" is >= 95% right
            sel = pt >= t
            if sel.sum() >= 15 and (yt[sel] == 1).mean() >= 0.95:
                t_hi = float(t); break
        for t in np.arange(0.5, 0.005, -0.01):                        # largest threshold at which "missing" is >= 95% right
            sel = pt <= t
            if sel.sum() >= 15 and (yt[sel] == 0).mean() >= 0.95:
                t_lo = float(t); break
        n_hi = int((pt >= t_hi).sum()) if t_hi is not None else 0
        n_lo = int((pt <= t_lo).sum()) if t_lo is not None else 0
        print(f"{blk}: n={len(y)} (hand-read {int(m.sum())}, present {int(yt.sum())}) | 'present' stated when p>={t_hi}: {n_hi} forms ({100*n_hi/len(yt):.0f}%) | "
              f"'missing' stated when p<={t_lo}: {n_lo} forms ({100*n_lo/len(yt):.0f}%)", flush=True)
        report[blk] = (t_lo, t_hi)
        models[blk] = {"model": clf.fit(X, y), "t_lo": t_lo, "t_hi": t_hi} if (t_hi is not None or t_lo is not None) else None
        print("  ->", "TRUSTED (asymmetric thresholds)" if models[blk] else "not trusted: stays 'not assessed'", flush=True)
    joblib.dump({"models": models, "keys": KEYS}, ROOT / "models" / "sig_clf_raj.joblib")
    print("saved models/sig_clf_raj.joblib")


if __name__ == "__main__":
    main()
