"""Whole-cell classifier for the Proforma-3 'affected area %' / 'crop loss %' cells.

python train_cells.py [--max-forms N] [--steps S] [--rebuild]
 * data: data/digits_cache/<docket>.npz (written by train_digits.build_cache / digits_locate) + app values in data/manifest.csv
   labels: row-1 area/loss cell <- app value (multiples of 5 in 0..100; anything else -> OTHER); row-2 cells (blank in
   practice) <- EMPTY.  Dockets in data/eval/cells.csv are never used for training; 10% of the other forms are a dev split.
 * model: small CNN on the line-removed ink mask, 64x192 -> 23 classes  (cells_cnn.npz, numpy inference in digits.py)
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

import digits as D

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "digits_cache")
DS = os.path.join(HERE, "data", "cells_ds.npz")
VALUES = list(range(0, 101, 5))
CELL_CLASSES = ["EMPTY"] + [str(v) for v in VALUES] + ["OTHER"]   # index 0 EMPTY, 1..21 values, 22 OTHER


def eval_dockets():
    p = os.path.join(HERE, "data", "eval")
    s = set()
    for f in glob.glob(os.path.join(p, "*.csv")):
        if os.path.basename(f) in ("app_values.csv", "mined_sel.csv"):
            continue  # app-wide tables, not hand labels
        try:
            d = pd.read_csv(f, dtype=str)
            if "docket" in d.columns:
                s |= set(d["docket"].dropna())
            elif "docket_id" in d.columns:
                s |= set(d["docket_id"].dropna())
        except Exception:
            pass
    return s


def label_of(v):
    if pd.isna(v):
        return None
    v = float(v)
    if v == int(v) and int(v) % 5 == 0 and 0 <= v <= 100:
        return 1 + int(v) // 5
    return len(CELL_CLASSES) - 1


def build_ds(max_forms=None):
    ev = eval_dockets()
    man = pd.read_csv(os.path.join(HERE, "data", "manifest.csv"), dtype={"docket_id": str}).drop_duplicates("docket_id").set_index("docket_id")
    # Human-QC'd values (written by retrain.py) override the app-entered values; the human hold-out is never trained on.
    hm = os.path.join(HERE, "data", "manifest_human.csv")
    if os.path.exists(hm):
        h = pd.read_csv(hm, dtype={"docket_id": str}).drop_duplicates("docket_id").set_index("docket_id")
        for d_, r_ in h.iterrows():
            if d_ not in man.index:
                man.loc[d_, :] = np.nan
            for col in ("affected_area_pct", "crop_loss_pct"):
                if pd.notna(r_.get(col)):
                    man.loc[d_, col] = r_[col]
    ho = os.path.join(HERE, "data", "holdout_human.csv")
    if os.path.exists(ho):
        ev = ev | set(pd.read_csv(ho, dtype=str)["docket_id"])
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE, "*.npz")))
    names = [n for n in names if n not in ev and n in man.index]
    if max_forms:
        names = names[:max_forms]
    X, y, dk, kind = [], [], [], []
    for n in names:
        try:
            z = np.load(os.path.join(CACHE, n + ".npz"))
            if not int(z["ok"]):
                continue
            for key, col in (("area", "affected_area_pct"), ("loss", "crop_loss_pct")):
                lab = label_of(man.loc[n, col])
                if lab is None or z[key].size < 500:
                    continue
                X.append(D.cell_image(z[key])); y.append(lab); dk.append(n); kind.append(0)
                k2 = "r2_" + key
                if k2 in z.files and z[k2].size > 500:
                    X.append(D.cell_image(z[k2])); y.append(0); dk.append(n); kind.append(1)
        except Exception:
            continue
    np.savez_compressed(DS, X=np.stack(X), y=np.array(y), dk=np.array(dk), kind=np.array(kind))
    print(f"dataset: {len(X)} cells from {len(set(dk))} forms; class counts", np.bincount(y, minlength=len(CELL_CLASSES)).tolist(), flush=True)


def make_net():
    import torch.nn as nn
    def blk(i, o):
        return [nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*blk(1, 16), *blk(16, 32), *blk(32, 48), *blk(48, 64), nn.Flatten(),
                         nn.Linear(64 * 4 * 12, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(CELL_CLASSES)))


def augment(x):
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    ang = (torch.rand(B) - .5) * 2 * np.radians(4)
    sh = (torch.rand(B) - .5) * 2 * 0.25
    sc = 0.88 + torch.rand(B) * 0.24
    sy = sc * (0.9 + torch.rand(B) * 0.2)
    tx, ty = (torch.rand(B) - .5) * 0.18, (torch.rand(B) - .5) * 0.22
    th = torch.zeros(B, 2, 3)
    cos, sin = torch.cos(ang), torch.sin(ang)
    th[:, 0, 0] = cos / sc; th[:, 0, 1] = (-sin * 0.33 + sh) / sc; th[:, 0, 2] = tx
    th[:, 1, 0] = sin * 3 / sy; th[:, 1, 1] = cos / sy; th[:, 1, 2] = ty
    x = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), align_corners=False)
    r = torch.rand(B, 1, 1, 1)
    x = torch.where(r < 0.3, F.max_pool2d(x, 3, 1, 1), torch.where(r > 0.85, -F.max_pool2d(-x, 3, 1, 1), x))
    x = torch.where(torch.rand(B, 1, 1, 1) < 0.3, F.avg_pool2d(x, 3, 1, 1), x)
    for i in torch.nonzero(torch.rand(B) < 0.25).flatten().tolist():  # ruled-line remnants
        if np.random.rand() < .5:
            y = np.random.choice([np.random.randint(0, 5), np.random.randint(59, 64)]); a = np.random.randint(0, 100)
            x[i, 0, y, a:a + np.random.randint(40, 192 - a + 1)] = 1.0
        else:
            xx = np.random.choice([np.random.randint(0, 4), np.random.randint(188, 192)]); x[i, 0, :, xx] = 1.0
    x = x + (torch.rand_like(x) < 0.004).float()
    return (x * (0.8 + 0.2 * torch.rand(B, 1, 1, 1))).clamp(0, 1)


def export(net, path):
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    names = {"0": "c1", "3": "c2", "6": "c3", "9": "c4", "13": "f1", "16": "f2"}
    arr = {}
    for k, n in names.items():
        arr[n + "w"] = sd[k + ".weight"]; arr[n + "b"] = sd[k + ".bias"]
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in arr.items()})


def main():
    import torch
    import torch.nn as nn
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-forms", type=int); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--rebuild", action="store_true"); ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--init", action="store_true", help="fine-tune from models/cells_cnn.npz instead of training from scratch")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.rebuild or not os.path.exists(DS):
        build_ds(a.max_forms)
    z = np.load(DS)
    X, y, dk = z["X"], z["y"], z["dk"]
    dev = np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 10 == 0 for d in dk])
    Xt = torch.tensor(X[~dev], dtype=torch.float32).div_(255).unsqueeze(1); yt = torch.tensor(y[~dev])
    Xd = torch.tensor(X[dev], dtype=torch.float32).div_(255).unsqueeze(1); yd = y[dev]
    cnt = np.bincount(y[~dev], minlength=len(CELL_CLASSES)).astype(float)
    wcls = 1.0 / np.sqrt(np.maximum(cnt, 1))          # sqrt-balanced sampling
    w = wcls[y[~dev]]
    hm = os.path.join(HERE, "data", "manifest_human.csv")   # human-checked forms (retrain.py): trusted labels -> 4x sampling weight
    if os.path.exists(hm):
        human = set(pd.read_csv(hm, dtype={"docket_id": str})["docket_id"])
        w = w * np.where(np.array([d in human for d in dk[~dev]]), 4.0, 1.0)
    p = torch.tensor(w / w.sum())
    net = make_net()
    mp = os.path.join(HERE, "models", "cells_cnn.npz")
    if a.init and os.path.exists(mp):
        zz = np.load(mp)
        sd = net.state_dict()
        for k, n in {"0": "c1", "3": "c2", "6": "c3", "9": "c4", "13": "f1", "16": "f2"}.items():
            sd[k + ".weight"] = torch.tensor(zz[n + "w"].astype(np.float32)); sd[k + ".bias"] = torch.tensor(zz[n + "b"].astype(np.float32))
        net.load_state_dict(sd)
        print("fine-tuning from the current model", flush=True)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=a.steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    t0 = time.time()
    for s in range(a.steps):
        net.train()
        idx = torch.multinomial(p, 64, replacement=True)
        opt.zero_grad(); l = lossf(net(augment(Xt[idx])), yt[idx]); l.backward(); opt.step(); sched.step()
        if (s + 1) % 500 == 0:
            net.eval()
            with torch.no_grad():
                pd_ = net(Xd).argmax(1).numpy()
            nz = yd > 1
            print(f"  step {s+1} loss {l.item():.3f} dev acc {np.mean(pd_==yd):.4f} nonzero-value acc {np.mean(pd_[nz]==yd[nz]):.4f} ({time.time()-t0:.0f}s)", flush=True)
            export(net, os.path.join(HERE, "models", "cells_cnn.npz"))
    export(net, os.path.join(HERE, "models", "cells_cnn.npz"))
    json.dump(dict(classes=CELL_CLASSES), open(os.path.join(HERE, "models", "cells_classes.json"), "w"))


if __name__ == "__main__":
    main()
