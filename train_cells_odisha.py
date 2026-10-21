"""Whole-cell classifier for the Odisha 'Crop Loss Joint Survey Form' Affected Area% / Loss% cells.

Same approach and same small CNN as train_cells.py (Proforma-3): classify the whole cell image into one of 23
classes (EMPTY, 0,5,..,100, OTHER) rather than reading individual digits - generic OCR cannot read this
handwriting reliably (confirmed against real samples), but a classifier trained on real labelled crops can.

  python tools/build_odisha_cache.py          # crops data/forms_odisha/*.pdf -> data/digits_cache_odisha/
  python train_cells_odisha.py --rebuild --steps 1500

data: data/digits_cache_odisha/<docket>.npz (area/loss raw gray crops) + data/odisha_manifest.csv (Excel ground
truth: Affected_Area_%, Crop_Loss_%). model: models/cells_cnn_odisha.npz (same numpy inference format as
cells_cnn.npz, loaded by digits.py's CELL_MODEL_PATH env override).
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
from train_cells import CELL_CLASSES, label_of, make_net, augment, export

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "data", "digits_cache_odisha")
DS = os.path.join(HERE, "data", "cells_ds_odisha.npz")
MODEL = os.path.join(HERE, "models", "cells_cnn_odisha.npz")


def build_ds():
    man = pd.read_csv(os.path.join(HERE, "data", "odisha_manifest.csv"), dtype={"docket_id": str}).drop_duplicates("docket_id").set_index("docket_id")
    names = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CACHE, "*.npz")))
    names = [n for n in names if n in man.index]
    X, y, dk, kind = [], [], [], []
    for n in names:
        try:
            z = np.load(os.path.join(CACHE, n + ".npz"))
            if not int(z["ok"]):
                continue
            for key, col in (("area", "affected_area_pct"), ("loss", "crop_loss_pct")):
                lab = label_of(man.loc[n, col])
                if lab is None or z[key].size < 200:
                    continue
                X.append(D.cell_image(z[key])); y.append(lab); dk.append(n); kind.append(0 if key == "area" else 1)
        except Exception:
            continue
    np.savez_compressed(DS, X=np.stack(X), y=np.array(y), dk=np.array(dk), kind=np.array(kind))
    print(f"dataset: {len(X)} cells from {len(set(dk))} forms; class counts", np.bincount(y, minlength=len(CELL_CLASSES)).tolist(), flush=True)


def main():
    import torch
    import torch.nn as nn
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--rebuild", action="store_true"); ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    if a.rebuild or not os.path.exists(DS):
        build_ds()
    z = np.load(DS)
    X, y, dk = z["X"], z["y"], z["dk"]
    dev = np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 10 == 0 for d in dk])  # 10% held-out forms
    Xt = torch.tensor(X[~dev], dtype=torch.float32).div_(255).unsqueeze(1); yt = torch.tensor(y[~dev])
    Xd = torch.tensor(X[dev], dtype=torch.float32).div_(255).unsqueeze(1); yd = y[dev]
    print(f"train: {len(yt)} cells ({(~dev).sum()} / {len(dk)} forms-rows), dev: {len(yd)} cells", flush=True)
    cnt = np.bincount(y[~dev], minlength=len(CELL_CLASSES)).astype(float)
    wcls = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = wcls[y[~dev]]
    p = torch.tensor(w / w.sum())
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=a.steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    t0 = time.time()
    for s in range(a.steps):
        net.train()
        idx = torch.multinomial(p, min(64, len(yt)), replacement=True)
        opt.zero_grad(); l = lossf(net(augment(Xt[idx])), yt[idx]); l.backward(); opt.step(); sched.step()
        if (s + 1) % 250 == 0:
            net.eval()
            with torch.no_grad():
                pd_ = net(Xd).argmax(1).numpy() if len(yd) else np.array([])
            nz = yd > 1
            acc = np.mean(pd_ == yd) if len(yd) else float("nan")
            nzacc = np.mean(pd_[nz] == yd[nz]) if nz.any() else float("nan")
            print(f"  step {s+1} loss {l.item():.3f} dev acc {acc:.4f} nonzero-value acc {nzacc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            export(net, MODEL)
    export(net, MODEL)
    json.dump(dict(classes=CELL_CLASSES), open(os.path.join(HERE, "models", "cells_classes_odisha.json"), "w"))
    print("saved", MODEL)


if __name__ == "__main__":
    main()
