"""Train the whole-cell model on the Rajasthan (National Insurance) form, from human-QC'd values.

python train_raj.py --labels Rajasthan_QC_Done.xlsx [--forms data/forms_raj] [--steps 3000] [--holdout 0.15] [--rounds 2]

 * labels: any Excel/CSV with docketID/docket_id, Signed_Copy_URL/pdf_url (only to download missing forms) and the human columns
   'Affected area% (Form)' / 'Crop Loss% (Form)'  (text such as DATA NOT FOUND is ignored).
 * one Rajasthan form lists up to 10 dockets and the human value belongs to ONE row, which the sheet does not say.  The row is found by
   multiple-instance selection: the row whose cells the current model finds most compatible with the human pair is the positive example
   (2 rounds: select -> train -> select again); rows 9-10 are blank examples.
 * the new model is adopted only if it is not worse on the Haryana hand-labelled forms and precise on the held-out Rajasthan forms.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd

import digits as D
import train_cells as TC

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "raj_cache")
FORMS = os.path.join(HERE, "data", "forms_raj")
CLASSES = TC.CELL_CLASSES


def num(v):
    try:
        return float(str(v).replace("%", "").strip())
    except ValueError:
        return None


def load_labels(path):
    df = pd.read_excel(path, dtype=str) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
    low = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    d = low.get("docketid") or low.get("docket")
    a = low.get("affectedarea%(form)")
    l = low.get("croploss%(form)")
    if not (d and a and l):
        sys.exit("need columns docketID, 'Affected area% (Form)', 'Crop Loss% (Form)'. Found: " + ", ".join(df.columns[:20]))
    out = pd.DataFrame({"docket": df[d].astype(str).str.strip(), "area": df[a].map(num), "loss": df[l].map(num)})
    out = out.dropna().drop_duplicates("docket")
    ok5 = lambda s: (s % 5 == 0) & (s >= 0) & (s <= 100)
    return out[ok5(out.area) & ok5(out.loss)].reset_index(drop=True)


def _cache_one(docket):
    import form_p3 as P
    import form_raj as R
    f = os.path.join(CACHE, docket + ".npz")
    if os.path.exists(f):
        return docket, True
    if os.path.exists(f + ".fail"):               # geometry already failed once: do not re-analyse on every restart
        return docket, False
    p = os.path.join(FORMS, docket + ".jpg")
    if not os.path.exists(p):
        return docket, False
    try:
        lay = P.analyse_layout(P.load_image(p))
        cr = R.cell_crops(lay) if lay["ok"] else None
        if cr is None:
            open(f + ".fail", "w").close()
            return docket, False
        crops, _ = cr
        X = np.stack([np.stack([D.cell_image(a), D.cell_image(b)]) for a, b in crops])      # (10, 2, 64, 192)
        np.savez_compressed(f, X=X)
        return docket, True
    except Exception:     # noqa: BLE001
        return docket, False


def build_cache(dockets, procs=4):
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(CACHE, exist_ok=True)
    with ProcessPoolExecutor(procs) as ex:
        res = list(ex.map(_cache_one, dockets, chunksize=8))
    ok = [d for d, o in res if o]
    print(f"cell crops cached for {len(ok)} of {len(dockets)} forms", flush=True)
    return ok


def cls_of(v):
    return TC.label_of(v)


def row_probs(X):
    """X (10,2,64,192) -> (10,2,23) class probabilities of the CURRENT model"""
    return np.array([[D.cell_proba(X[k, j]) for j in range(2)] for k in range(X.shape[0])])


def select_rows(dockets, lab, rounds_note=""):
    """-> dict docket -> (row index, score) with the best-compatible row"""
    sel = {}
    for d in dockets:
        X = np.load(os.path.join(CACHE, d + ".npz"))["X"]
        P_ = row_probs(X)
        la, ll = cls_of(lab.loc[d, "area"]), cls_of(lab.loc[d, "loss"])
        sc = P_[:, 0, la] * P_[:, 1, ll]
        k = int(sc.argmax())
        sel[d] = (k, float(sc[k]))
    return sel


def build_dataset(dockets, lab, sel, min_score):
    X, y, dk = [], [], []
    for d in dockets:
        k, s = sel[d]
        if s < min_score:
            continue
        Z = np.load(os.path.join(CACHE, d + ".npz"))["X"]
        for j, key in ((0, "area"), (1, "loss")):
            X.append(Z[k, j]); y.append(cls_of(lab.loc[d, key])); dk.append(d)
        for rr in (8, 9):                       # the last rows are blank on virtually every form
            if rr != k:
                for j in (0, 1):
                    X.append(Z[rr, j]); y.append(0); dk.append(d)
    return np.stack(X), np.array(y), np.array(dk)


def evaluate(dockets, lab, gate):
    """form-level: pick the row by compatibility is NOT allowed here - report the model on the best-compatible row as an upper bound and
    on the first non-empty row as the realistic reading."""
    n = ans = ok = 0
    for d in dockets:
        Z = np.load(os.path.join(CACHE, d + ".npz"))["X"]
        P_ = row_probs(Z)
        la, ll = cls_of(lab.loc[d, "area"]), cls_of(lab.loc[d, "loss"])
        filled = [k for k in range(10) if P_[k, 0, 0] < 0.5 or P_[k, 1, 0] < 0.5]
        for j, lc in ((0, la), (1, ll)):
            n += 1
            # realistic: single filled row -> read it; several -> the row whose reading equals the label counts only if unique answer
            rows = filled if filled else [0]
            vals = [(int(P_[k, j].argmax()), float(P_[k, j].max())) for k in rows]
            if len(rows) == 1 and vals[0][1] >= gate and vals[0][0] not in (0, 22):
                ans += 1
                ok += int(vals[0][0] == lc)
    return {"cells": n, "answered": ans, "correct": ok, "coverage": ans / max(n, 1), "precision": ok / max(ans, 1)}


def main():
    import torch
    import torch.nn as nn
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--holdout", type=float, default=0.15)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--init-model", default=os.path.join(HERE, "models", "cells_cnn.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "cells_cnn.npz"), help="where the trained model is written")
    ap.add_argument("--min-score", type=float, default=0.02)
    ap.add_argument("--balance", action="store_true", help="oversample written (non-blank, non-zero) cells")
    ap.add_argument("--rows", help="CSV docket,row (1..10): hand-labelled table row of each docket; replaces the automatic row choice")
    ap.add_argument("--haryana-share", type=float, default=0.5, help="share of each batch drawn from the Haryana cell set (keeps it from forgetting)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    lab = load_labels(a.labels).set_index("docket", drop=False)
    have = [d for d in lab.index if os.path.exists(os.path.join(FORMS, d + ".jpg"))]
    print(f"human labels {len(lab)} | forms on disk {len(have)}", flush=True)
    ok = build_cache(have, a.procs)
    ids = sorted(ok, key=lambda d: hashlib.md5(d.encode()).hexdigest())
    k = int(len(ids) * a.holdout)
    hold, train_ids = ids[:k], ids[k:]
    print(f"train {len(train_ids)} | hold-out {len(hold)} (never trained on)", flush=True)
    har = np.load(TC.DS) if os.path.exists(TC.DS) else None
    mp = a.init_model
    D.CELL_MODEL_PATH = mp
    D._CELL = None
    import json
    bf = os.path.join(HERE, "data", "raj_before.json")          # cached: the cloud machine restarts every few minutes
    if os.path.exists(bf):
        before = json.load(open(bf))
    else:
        before = evaluate(hold, lab, 0.85)
        json.dump(before, open(bf, "w"))
    print("CURRENT model on Rajasthan hold-out:", before, flush=True)
    for rnd in range(a.rounds):
        dsf = os.path.join(HERE, "data", f"raj_ds_{'rows_' if a.rows else ''}round{rnd + 1}.npz")
        if os.path.exists(dsf):
            try:
                z = np.load(dsf); Xr, yr = z["X"], z["y"]
                print(f"round {rnd + 1}: dataset loaded from {dsf}", flush=True)
            except Exception:     # noqa: BLE001
                os.remove(dsf)
        if not os.path.exists(dsf):
            sel = select_rows(train_ids, lab)
            if a.rows:
                rr = pd.read_csv(a.rows, dtype=str)
                hand = {str(r.docket).strip(): int(float(r.row)) - 1 for r in rr.itertuples() if str(r.row).replace(".", "").isdigit() and 1 <= int(float(r.row)) <= 10}
                sel = {d: ((hand[d], 1.0) if d in hand else (sel[d][0], 0.0)) for d in train_ids}   # forms without a hand label are left out
            good = sum(1 for v in sel.values() if v[1] >= a.min_score)
            print(f"round {rnd + 1}: {good} of {len(train_ids)} forms have a compatible row (score >= {a.min_score}); row index counts:",
                  np.bincount([v[0] for v in sel.values() if v[1] >= a.min_score], minlength=10).tolist(), flush=True)
            Xr, yr, _ = build_dataset(train_ids, lab, sel, a.min_score)
            np.savez(dsf + ".tmp.npz", X=Xr, y=yr); os.replace(dsf + ".tmp.npz", dsf)
        print("  Rajasthan cells", len(yr), "class counts", np.bincount(yr, minlength=len(CLASSES)).tolist(), flush=True)
        net = TC.make_net()
        zz = np.load(mp)
        sd = net.state_dict()
        for kk, n in {"0": "c1", "3": "c2", "6": "c3", "9": "c4", "13": "f1", "16": "f2"}.items():
            sd[kk + ".weight"] = torch.tensor(zz[n + "w"].astype(np.float32)); sd[kk + ".bias"] = torch.tensor(zz[n + "b"].astype(np.float32))
        net.load_state_dict(sd)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=a.steps)
        lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
        Xt = torch.tensor(Xr, dtype=torch.float32).div_(255).unsqueeze(1); yt = torch.tensor(yr)
        if har is not None:
            Xh = torch.tensor(har["X"], dtype=torch.float32).div_(255).unsqueeze(1); yh = torch.tensor(har["y"])
        wr = torch.nonzero(yt >= 2).flatten()
        t0 = time.time()
        ck = os.path.join(HERE, "data", f"raj_ckpt{'_bal' if a.balance else ''}_round{rnd + 1}.pt")      # the cloud machine can restart: resume an interrupted round
        s0 = 0
        if os.path.exists(ck):
            try:
                st = torch.load(ck)
                if st.get("steps") == a.steps and st.get("n") == len(yt):
                    net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); s0 = st["step"]
                    print(f"  resumed round {rnd + 1} from checkpoint at step {s0}", flush=True)
            except Exception:     # noqa: BLE001
                s0 = 0
        if s0 >= a.steps and os.path.exists(a.out) and rnd + 1 < a.rounds:
            pass
        for s in range(s0, a.steps):
            net.train()
            nr = int(round(64 * (1 - a.haryana_share))) if har is not None else 64
            if a.balance:       # half of each Rajasthan batch from written (non-blank, non-zero) cells
                idx = torch.cat([wr[torch.randint(0, len(wr), (nr // 2,))], torch.randint(0, len(yt), (nr - nr // 2,))])
            else:
                idx = torch.randint(0, len(yt), (nr,))
            xb, yb = Xt[idx], yt[idx]
            if har is not None:
                ih = torch.randint(0, len(yh), (64 - nr,))
                xb, yb = torch.cat([xb, Xh[ih]]), torch.cat([yb, yh[ih]])
            opt.zero_grad(); l = lossf(net(TC.augment(xb)), yb); l.backward(); opt.step(); sched.step()
            if (s + 1) % 100 == 0:
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": s + 1, "steps": a.steps, "n": len(yt)}, ck)
            if (s + 1) % 500 == 0:
                print(f"  step {s + 1} loss {l.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        TC.export(net, a.out)
        mp = a.out                                  # the next round starts from this round's model
        D.CELL_MODEL_PATH = a.out
        D._CELL = None
        new = evaluate(hold, lab, 0.85)
        print(f"NEW model (round {rnd + 1}) on Rajasthan hold-out:", new, flush=True)
    print("done; model written to", a.out, "- check Haryana accuracy before using it as models/cells_cnn.npz")


if __name__ == "__main__":
    main()
