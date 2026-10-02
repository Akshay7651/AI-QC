#!/usr/bin/env python
"""Train the digit-by-digit area/loss cell reader -> models/cellseq_crnn.npz  (inference: cellseq_reader.py).

  python train_cellseq.py [--epochs 10] [--synth 8000]
  python train_cellseq.py --eval          # hold-out only (data/holdout_human.csv: never trained on)

Data: the cached row-1 cell crops (data/digits_cache/<docket>.npz: 'area', 'loss') with the app values of
data/manifest_human.csv as weak labels. How a value was WRITTEN is unknown ("0" / "00" / "0%" / "90%"), so the CTC loss is the
marginal likelihood over those writing variants (same idea as the date reader). Synthetic handwritten numbers (MNIST digits)
are mixed in so the reader looks at the digits instead of guessing from the most common values.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import date_reader as DR        # noqa: E402
import train_dates as TD        # noqa: E402
import cellseq_reader as CR     # noqa: E402

PCT = 11                         # class id of '%' (the date model's separator class)
OUT = CR.MODEL_PATH


def variants(v):
    s = str(int(v))
    base = [s] + (["00", "000"] if v == 0 else []) + (["0" + s] if 0 < v < 10 else [])
    out = []
    for b in base:
        t = [int(c) + 1 for c in b]
        out += [t, t + [PCT]]
    return out


def load():
    man = pd.read_csv(os.path.join(ROOT, "data", "manifest_human.csv"), dtype=str).set_index("docket_id")
    hold = set(pd.read_csv(os.path.join(ROOT, "data", "holdout_human.csv"), dtype=str).docket_id)
    tr, ho = [], []
    for d, r in man.iterrows():
        f = os.path.join(ROOT, "data", "digits_cache", d + ".npz")
        if not os.path.exists(f):
            continue
        try:
            z = np.load(f)
            if not int(z["ok"]):
                continue
        except Exception:     # noqa: BLE001
            continue
        for key, col in (("area", "affected_area_pct"), ("loss", "crop_loss_pct")):
            try:
                v = float(r[col])
            except (TypeError, ValueError):
                continue
            if not (0 <= v <= 100) or v != int(v) or z[key].size < 500:
                continue
            it = {"docket": d, "crop": z[key], "value": v, "vars": variants(v)}
            (ho if d in hold else tr).append(it)
    return tr, ho


def synth_items(n, seed=0):
    rng = np.random.default_rng(seed)
    vals = list(range(0, 101, 5)) + list(range(0, 101))
    out = []
    for _ in range(n):
        v = int(rng.choice(vals))
        s = ("00" if (v == 0 and rng.random() < 0.4) else str(v))
        out.append({"docket": "synth", "text": s, "vars": [[int(c) + 1 for c in s]], "synthetic": True, "value": float(v)})
    return out


class DS:
    def __init__(self, items, aug=True):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        it = self.items[i]
        rng = np.random.default_rng((i * 7919 + int(time.time() * 1000)) % (2 ** 32))
        if it.get("synthetic"):
            g = TD.augment(TD.render_synth(it, rng), rng, trim=False)
        else:
            g = TD.augment(it["crop"], rng, trim=True) if self.aug else it["crop"]
        return torch.from_numpy(CR.prep(g))[None], i


def ctc_marginal(net_out, idxs, items):
    import torch
    import torch.nn.functional as F
    lp = F.log_softmax(net_out, -1).permute(1, 0, 2)
    T = lp.shape[0]
    tg, lens, owner = [], [], []
    for bi, i in enumerate(idxs):
        for v in items[i]["vars"]:
            tg += v; lens.append(len(v)); owner.append(bi)
    ow = torch.tensor(owner)
    nll = F.ctc_loss(lp[:, ow], torch.tensor(tg), torch.full((len(owner),), T, dtype=torch.long), torch.tensor(lens),
                     blank=0, reduction="none", zero_infinity=True)
    M = torch.full((len(idxs), len(owner)), float("inf"))
    M[ow, torch.arange(len(owner))] = nll
    return -torch.logsumexp(-M, 1)


def train(net, items, epochs, lr, bs=64):
    import torch
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(items) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.15)
    dl = torch.utils.data.DataLoader(DS(items), batch_size=bs, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    for ep in range(epochs):
        net.train(); t0 = time.time(); tot = n = 0
        for x, idx in dl:
            loss = ctc_marginal(net(x), idx.tolist(), items).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); n += len(idx)
        print("epoch %d/%d loss %.3f  %.0fs" % (ep + 1, epochs, tot / max(n, 1), time.time() - t0), flush=True)
        TD.export(net, OUT.replace(".npz", "_partial.npz"), {"kind": "cellseq"})
    return net


def evaluate(items, model=OUT):
    CR.MODEL_PATH = model; CR._M = None
    rows = [(it["value"], CR.read_value(it["crop"])) for it in items]
    for nm, sel in (("non-zero", lambda v: v > 0), ("zero", lambda v: v == 0)):
        R = [(v, r) for v, r in rows if sel(v)]
        for g in (0.5, 0.7, 0.8, 0.9, 0.95):
            ans = [(v, r) for v, r in R if r["value"] is not None and r["conf"] >= g]
            ok = sum(abs(r["value"] - v) < 1e-6 for v, r in ans)
            print(f"{nm:8s} gate {g:.2f}: answered {len(ans)}/{len(R)} ({len(ans) / max(1, len(R)):.0%}) right {ok / max(1, len(ans)):.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--synth", type=int, default=8000)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    tr, ho = load()
    print(f"cells: train {len(tr)} | hold-out {len(ho)} (never trained on) | non-zero in train {sum(x['value'] > 0 for x in tr)}", flush=True)
    if a.eval:
        evaluate(ho)
        return
    items = tr + synth_items(a.synth)
    net = TD.make_model()
    train(net, items, a.epochs, a.lr)
    TD.export(net, OUT, {"kind": "cellseq"})
    print("saved", OUT, flush=True)
    evaluate(ho)


if __name__ == "__main__":
    main()
