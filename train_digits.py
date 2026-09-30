"""Train the Proforma-3 digit model.  Usage:  python train_digits.py [--quick] [--rounds 2] [--max-forms N]

Pipeline (all offline):
 1. data/mnist/*  (downloaded automatically from the cvdf-datasets mirror if missing)
 2. every data/forms/*.jpg is located with digits_locate.py (table cells + PO-ID strip) and cached in data/digits_cache/
 3. round 0: CNN on MNIST + synthetic junk/mark shapes.  Then harvest REAL labelled components:
      - PO-ID strip: 18 handwritten digits == the docket id (form kept only if >= MIN_AGREE/18 positions agree
        with the current model => filters mis-located strips)
      - table cells: weak labels from data/manifest.csv (affected_area_pct / crop_loss_pct), kept only where the
        current model's reading of the leading digits agrees; trailing components become the 'mark' class (% or slash)
    and retrain on MNIST + real (forms are split 80/20 by docket hash; held-out forms are never used for training).
 4. writes models/digits_cnn.npz (float16) and models/metrics.json, prints held-out metrics.
"""
import argparse
import glob
import gzip
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

import digits as D
import digits_locate as L

HERE = os.path.dirname(os.path.abspath(__file__))
MN = os.path.join(HERE, "data", "mnist")
CACHE = os.path.join(HERE, "data", "digits_cache")
MIN_AGREE = {0: 11, 1: 13, 2: 14}


def is_test_form(docket):
    return int(hashlib.md5(str(docket).encode()).hexdigest(), 16) % 5 == 0


# ------------------------------------------------------------------ MNIST
def load_mnist():
    os.makedirs(MN, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    out = {}
    for k in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        p = os.path.join(MN, k + ".gz")
        if not os.path.exists(p):
            urllib.request.urlretrieve(base + k + ".gz", p)
        with gzip.open(p) as f:
            d = f.read()
        out[k] = (np.frombuffer(d, np.uint8, offset=16).reshape(-1, 28, 28) if "images" in k else np.frombuffer(d, np.uint8, offset=8))
    return out["train-images-idx3-ubyte"], out["train-labels-idx1-ubyte"], out["t10k-images-idx3-ubyte"], out["t10k-labels-idx1-ubyte"]


# ------------------------------------------------------------------ crop cache
def _cache_one(path):
    dk = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(CACHE, dk + ".npz")
    if os.path.exists(out):
        return dk, True
    try:
        img = Image.open(path)
        img.draft('L', (img.width // 2, img.height // 2)) if img.width >= 3200 else None
        t = L.locate_table(img)
        if t is None:
            np.savez_compressed(out, ok=np.array(0))
            return dk, False
        cc = L.cell_crops(img)
        strip = L.po_strip(t)
        np.savez_compressed(out, ok=np.array(1), strip=strip, area=np.asarray(cc["area"]), loss=np.asarray(cc["loss"]),
                            t_area=np.asarray(cc.get("t_area", np.zeros((1, 1), np.uint8))),
                            t_loss=np.asarray(cc.get("t_loss", np.zeros((1, 1), np.uint8))),
                            has_total=np.array(int("t_area" in cc)))
        return dk, True
    except Exception as e:  # noqa
        np.savez_compressed(out, ok=np.array(0))
        return dk, False


def build_cache(max_forms=None, procs=4):
    os.makedirs(CACHE, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(HERE, "data", "forms", "*.jpg")))
    if max_forms:
        fs = fs[:max_forms]
    todo = [f for f in fs if not os.path.exists(os.path.join(CACHE, os.path.basename(f)[:-4] + ".npz"))]
    if todo:
        t0 = time.time()
        with Pool(procs) as p:
            for i, _ in enumerate(p.imap_unordered(_cache_one, todo, chunksize=4)):
                if i % 50 == 0:
                    print(f"  cached {i}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    names = [os.path.basename(f)[:-4] for f in fs]
    return [n for n in names if os.path.exists(os.path.join(CACHE, n + ".npz")) and int(np.load(os.path.join(CACHE, n + ".npz"))["ok"])]


# ------------------------------------------------------------------ synthetic shapes
def synth_shapes(n, rng, kind):
    """kind 'mark' (slash-like strokes) or 'junk' (dashes, blobs, arcs) -> (n,28,28) uint8 already normalised like norm28."""
    import cv2
    out = np.zeros((n, 28, 28), np.uint8)
    for i in range(n):
        c = np.zeros((28, 28), np.uint8)
        th = int(rng.integers(1, 4))
        if kind == "mark":
            t = rng.integers(0, 4)
            if t <= 1:  # slash
                ang = np.radians(rng.uniform(25, 65))
                ln = rng.uniform(16, 21)
                dx, dy = np.cos(ang) * ln / 2, np.sin(ang) * ln / 2
                p1 = (14 - dx + rng.normal(0, .7), 14 + dy + rng.normal(0, .7)); p2 = (14 + dx + rng.normal(0, .7), 14 - dy + rng.normal(0, .7))
                cv2.line(c, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 255, th)
            elif t == 2:  # x / cross
                for s in (1, -1):
                    cv2.line(c, (5, 14 - 6 * s), (22, 14 + 6 * s), 255, th)
            else:  # slash with a curl / dot pair (o/o glyph squeezed)
                cv2.line(c, (8, 22), (20, 6), 255, th)
                cv2.circle(c, (8, 8), 3, 255, 1); cv2.circle(c, (20, 21), 3, 255, 1)
        else:
            t = rng.integers(0, 4)
            if t == 0:  # horizontal dash, slightly tilted
                y = int(rng.integers(10, 18)); cv2.line(c, (4, y), (24, y + int(rng.integers(-3, 4))), 255, th)
            elif t == 1:  # blob
                cv2.ellipse(c, (14, 14), (int(rng.integers(3, 9)), int(rng.integers(3, 9))), 0, 0, 360, 255, -1)
            elif t == 2:  # partial arc
                a0 = rng.uniform(0, 360); cv2.ellipse(c, (14, 14), (9, int(rng.integers(6, 10))), 0, a0, a0 + rng.uniform(60, 160), 255, th)
            else:  # vertical tick / short stroke
                x = int(rng.integers(10, 18)); cv2.line(c, (x, 4), (x + int(rng.integers(-2, 3)), 24), 255, th)
        out[i] = c
    return out


# ------------------------------------------------------------------ torch model + training
def make_net():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 7 * 7, 96), nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, D.NCLS))


def augment(x, strong=True):
    """x (B,1,28,28) float tensor -> augmented."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    ang = (torch.rand(B) - .5) * 2 * np.radians(14)
    sh = (torch.rand(B) - .5) * 2 * 0.30
    sc = 0.85 + torch.rand(B) * 0.3
    sy = sc * (0.9 + torch.rand(B) * 0.2)
    tx, ty = (torch.rand(B) - .5) * 0.16, (torch.rand(B) - .5) * 0.16
    cos, sin = torch.cos(ang), torch.sin(ang)
    th = torch.zeros(B, 2, 3)
    th[:, 0, 0] = cos / sc; th[:, 0, 1] = (-sin + sh) / sc; th[:, 0, 2] = tx
    th[:, 1, 0] = sin / sy; th[:, 1, 1] = cos / sy; th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False)
    r = torch.rand(B, 1, 1, 1)
    thick = F.max_pool2d(x, 3, 1, 1)
    thin = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(r < 0.35, thick, torch.where(r > 0.85, thin, x))
    # blur some
    bl = F.avg_pool2d(x, 3, 1, 1)
    x = torch.where(torch.rand(B, 1, 1, 1) < 0.3, bl, x)
    # line remnants: horizontal/vertical streak
    if strong:
        m = (torch.rand(B) < 0.15)
        for i in torch.nonzero(m).flatten().tolist():
            if np.random.rand() < 0.5:
                y = np.random.randint(2, 26); x0 = np.random.randint(0, 8); x[i, 0, y, x0:x0 + np.random.randint(6, 28)] = 1.0
            else:
                xx = np.random.randint(0, 3) * 12 + np.random.randint(0, 3); x[i, 0, :, xx] = torch.maximum(x[i, 0, :, xx], torch.full((28,), 0.8))
        x = x + (torch.rand_like(x) < 0.01).float() * torch.rand_like(x)  # salt
    x = x * (0.75 + 0.25 * torch.rand(B, 1, 1, 1))
    return x.clamp(0, 1)


def train_net(Xm, ym, Xr, yr, epochs, seed=0, real_w=6, log=True):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(4)
    rng = np.random.default_rng(seed)
    # MNIST digits: resize like norm28 would (digit occupies 20px box, centred by COM) -> already ~that
    Xs_m = synth_shapes(6000, rng, "mark"); Xs_j = synth_shapes(6000, rng, "junk")
    X = np.concatenate([Xm, Xs_m, Xs_j]); y = np.concatenate([ym, np.full(len(Xs_m), 11), np.full(len(Xs_j), 12)])
    w = np.ones(len(X), np.float32)
    if len(Xr):
        X = np.concatenate([X, Xr]); y = np.concatenate([y, yr]); w = np.concatenate([w, np.full(len(Xr), real_w, np.float32)])
    Xt = torch.tensor(X, dtype=torch.float32).div_(255 if X.dtype == np.uint8 else 1).unsqueeze(1)
    yt = torch.tensor(y, dtype=torch.long)
    p = torch.tensor(w / w.sum())
    net = make_net()
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    steps_per = 600
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=epochs * steps_per)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    net.train()
    for ep in range(epochs):
        tl = 0
        for s in range(steps_per):
            idx = torch.multinomial(p, 128, replacement=True)
            xb = augment(Xt[idx]); yb = yt[idx]
            opt.zero_grad(); l = lossf(net(xb), yb); l.backward(); opt.step(); sched.step(); tl += l.item()
        if log:
            print(f"    epoch {ep+1}/{epochs} loss {tl/steps_per:.4f}", flush=True)
    net.eval()
    return net


def export(net, path):
    sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    arr = dict(c1w=sd["0.weight"], c1b=sd["0.bias"], c2w=sd["3.weight"], c2b=sd["3.bias"],
               f1w=sd["7.weight"], f1b=sd["7.bias"], f2w=sd["10.weight"], f2b=sd["10.bias"])
    np.savez_compressed(path, **{k: v.astype(np.float16) for k, v in arr.items()})


# ------------------------------------------------------------------ harvesting real components
def _load_cache(dk):
    z = np.load(os.path.join(CACHE, dk + ".npz"))
    return {k: z[k] for k in z.files}


def po_components(strip):
    """PO-ID strip -> the 18 handwritten digit components (last 18 of consistent height) or None."""
    comps, H, W = D._analyse(strip)
    comps = [c for c in comps if not c["dot"] and (c["y1"] - c["y0"]) >= 0.30 * H]
    if len(comps) < 18:
        return None
    hs = np.array([c["y1"] - c["y0"] for c in comps])
    comps = comps[-18:]
    return comps


def fmt_label(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = float(v)
    return str(int(v)) if v == int(v) else ("%g" % v)


def harvest(forms, man, rnd, log=True):
    """-> X (n,28,28 uint8), y, meta(list of (docket,src)); uses current model (digits.MODEL_PATH)."""
    Xs, ys, meta = [], [], []
    n_po = n_po_ok = 0
    n_cells = n_cells_ok = 0
    for dk in forms:
        if is_test_form(dk):
            continue
        z = _load_cache(dk)
        # ---- PO id
        if len(dk) == 18:
            comps = po_components(z["strip"])
            n_po += 1
            if comps is not None:
                pr = D.classify_components(comps)
                pred = pr[:, :10].argmax(1)
                agree = int((pred == np.array([int(c) for c in dk])).sum())
                if agree >= MIN_AGREE.get(rnd, 14):
                    n_po_ok += 1
                    for c, ch in zip(comps, dk):
                        Xs.append((D.norm28(c) * 255).astype(np.uint8)); ys.append(int(ch)); meta.append((dk, "po"))
        # ---- cells
        if dk in man.index:
            r = man.loc[dk]
            for field, keys in (("area", ("area", "t_area")), ("loss", ("loss", "t_loss"))):
                lab = fmt_label(r["affected_area_pct" if field == "area" else "crop_loss_pct"])
                if lab is None:
                    continue
                for key in keys:
                    cell = z[key]
                    if cell.size < 100:
                        continue
                    comps, H, W = D._analyse(cell)
                    comps = [c for c in comps if not c["dot"]]
                    if not comps:
                        continue
                    n_cells += 1
                    pr = D.classify_components(comps)
                    digs = [ch for ch in lab if ch.isdigit()]
                    if "." in lab:
                        continue  # decimals handled by rule; skip weak labelling
                    need = len(digs)
                    pred = pr.argmax(1)
                    # row of a value '0' may be written '0' or '00'
                    if lab == "0":
                        cand = [i for i, c in enumerate(comps) if pred[i] == 0 and pr[i, 0] > 0.5]
                        if cand and cand[0] == 0 and len(comps) <= 4:
                            tgt = [(0, 0)]
                            if len(comps) > 1 and pred[1] == 0 and pr[1, 0] > 0.5:
                                tgt.append((1, 0))
                            rest = [i for i in range(len(comps)) if i not in [t[0] for t in tgt]]
                        else:
                            continue
                    else:
                        if len(comps) < need:
                            continue
                        ok = all(pred[i] == int(digs[i]) and pr[i, int(digs[i])] > 0.3 for i in range(need))
                        if not ok:
                            continue
                        tgt = [(i, int(digs[i])) for i in range(need)]
                        rest = list(range(need, len(comps)))
                    n_cells_ok += 1
                    for i, cl in tgt:
                        Xs.append((D.norm28(comps[i]) * 255).astype(np.uint8)); ys.append(cl); meta.append((dk, "cell"))
                    for i in rest:  # trailing comps: '%' / slash / pen marks
                        h = comps[i]["y1"] - comps[i]["y0"]
                        if h > 0.3 * H:
                            Xs.append((D.norm28(comps[i]) * 255).astype(np.uint8)); ys.append(11); meta.append((dk, "cell"))
    if log:
        print(f"  harvest round {rnd}: PO strips accepted {n_po_ok}/{n_po}; cells accepted {n_cells_ok}/{n_cells}; comps {len(Xs)}", flush=True)
    return (np.stack(Xs) if Xs else np.zeros((0, 28, 28), np.uint8)), np.array(ys, np.int64), meta


# ------------------------------------------------------------------ evaluation
def evaluate(forms, man, log=True):
    res = {}
    # PO id
    tot_d = ok_d = ex = n = 0
    n_seg = 0
    for dk in forms:
        if not is_test_form(dk) or len(dk) != 18:
            continue
        z = _load_cache(dk)
        n += 1
        comps = po_components(z["strip"])
        if comps is None:
            continue
        n_seg += 1
        pr = D.classify_components(comps)[:, :10].argmax(1)
        a = int((pr == np.array([int(c) for c in dk])).sum())
        ok_d += a; tot_d += 18; ex += int(a == 18)
    res["po_heldout_forms"] = n
    res["po_segmented_18"] = n_seg
    res["po_digit_acc_on_segmented"] = round(ok_d / max(tot_d, 1), 4)
    res["po_exact_18_of_segmented"] = round(ex / max(n_seg, 1), 4)
    res["po_exact_18_of_all"] = round(ex / max(n, 1), 4)
    # weak-label cells (app values), held-out forms only, pipeline end-to-end
    c_ok = c_n = 0
    for dk in forms:
        if not is_test_form(dk) or dk not in man.index:
            continue
        z = _load_cache(dk); r = man.loc[dk]
        for f, key, col in (("area", "area", "affected_area_pct"), ("loss", "loss", "crop_loss_pct")):
            v = r[col]
            if pd.isna(v):
                continue
            out = D.read_number(Image.fromarray(z[key]))
            c_n += 1; c_ok += int(out["value"] is not None and abs(out["value"] - float(v)) < 1e-6)
    res["weak_cell_n"] = c_n
    res["weak_cell_exact"] = round(c_ok / max(c_n, 1), 4)
    if log:
        print("  eval:", json.dumps(res), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-forms", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_mnist()
    # MNIST digits re-normalised exactly like real components (tight bbox -> 20px box, COM centred)
    def mn_norm(X):
        out = np.zeros_like(X)
        for i, im in enumerate(X):
            ys_, xs_ = np.nonzero(im > 40)
            if len(ys_) == 0:
                continue
            m = im[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
            out[i] = (D.norm28(dict(mask=m)) * 255).astype(np.uint8)
        return out
    Xtr, Xte = mn_norm(Xtr), mn_norm(Xte)
    print(f"MNIST normalised ({time.time()-t0:.0f}s)", flush=True)
    forms = build_cache(a.max_forms)
    print(f"{len(forms)} forms with located table; test forms {sum(is_test_form(f) for f in forms)}", flush=True)
    man = pd.read_csv(os.path.join(HERE, "data", "manifest.csv"), dtype={"docket_id": str}).drop_duplicates("docket_id").set_index("docket_id")
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    ep = a.epochs or (3 if a.quick else 8)
    Xr = np.zeros((0, 28, 28), np.uint8); yr = np.zeros(0, np.int64)
    net = None
    for rnd in range(a.rounds):
        print(f"round {rnd}: training ({len(Xr)} real comps)", flush=True)
        net = train_net(Xtr, ytr, Xr, yr, ep if rnd < a.rounds - 1 else ep + 4, seed=rnd)
        export(net, D.MODEL_PATH); D._MODEL = None
        import torch
        with torch.no_grad():
            acc = (net(torch.tensor(Xte, dtype=torch.float32).div(255).unsqueeze(1)).argmax(1).numpy() == yte).mean()
        print(f"  MNIST-test acc {acc:.4f}", flush=True)
        if rnd < a.rounds - 1:
            Xr, yr, _ = harvest(forms, man, rnd + 1 if rnd + 1 in MIN_AGREE else 2)
    res = evaluate(forms, man)
    res["mnist_test_acc"] = float(acc); res["n_real_comps"] = int(len(Xr)); res["n_forms"] = len(forms)
    json.dump(res, open(os.path.join(HERE, "models", "metrics.json"), "w"), indent=1)
    print(f"done in {time.time()-t0:.0f}s; model {os.path.getsize(D.MODEL_PATH)/1e3:.0f} kB", flush=True)


if __name__ == "__main__":
    main()
