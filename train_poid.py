#!/usr/bin/env python
"""Train the handwritten PO-ID reader (same CNN + BiGRU + CTC as the date reader) -> models/poid_crnn.npz.

  python tools/extract_poid_crops.py --limit 6000     # once: data/poid_crops.pkl
  python train_poid.py [--epochs 12] [--round2]       # --round2 drops the samples the model disagrees with most (forms where
                                                      #   the surveyor wrote another number, e.g. the policy no.) and fine-tunes
  python train_poid.py --eval                         # exact / last-8-digit accuracy on the 10% hold-out (never trained on)

Weak labels: the docket id of each form (the surveyor is supposed to write it as PO ID). Inference: poid_reader.read_po_id().
"""
import argparse
import hashlib
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import date_reader as DR      # noqa: E402  (preprocessing + numpy forward pass are shared)
import train_dates as TD      # noqa: E402  (model, augmentation, export)
import poid_reader as PR      # noqa: E402

CROPS = os.path.join(ROOT, "data", "poid_crops.pkl")
OUT = os.path.join(ROOT, "models", "poid_crnn.npz")


def holdout(d):
    return int(hashlib.md5(d.encode()).hexdigest(), 16) % 10 == 0


def load():
    z = pickle.load(open(CROPS, "rb"))
    items = []
    for it in z:
        g = it["crop"]
        if g is None or g.shape[1] / max(1, g.shape[0]) > 25 or g.shape[1] < 150:      # failed / implausible crops
            continue
        d = it["docket"]
        if len(d) != 18 or not d.isdigit():
            continue
        items.append({"docket": d, "crop": g, "tgt": [int(c) + 1 for c in d]})
    tr = [x for x in items if not holdout(x["docket"])]
    ho = [x for x in items if holdout(x["docket"])]
    return tr, ho


class DS:
    def __init__(self, items, aug=True):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        it = self.items[i]
        rng = np.random.default_rng((i * 7919 + int(time.time() * 1000)) % (2 ** 32))
        g = TD.augment(it["crop"], rng, trim=False) if self.aug else it["crop"]
        return torch.from_numpy(PR.prep(g))[None], i


def ctc(net_out, idxs, items):
    import torch
    import torch.nn.functional as F
    lp = F.log_softmax(net_out, -1).permute(1, 0, 2)
    T = lp.shape[0]
    tg = [items[i]["tgt"] for i in idxs]
    return F.ctc_loss(lp, torch.tensor([c for t in tg for c in t]), torch.full((len(idxs),), T, dtype=torch.long),
                      torch.tensor([len(t) for t in tg]), blank=0, reduction="none", zero_infinity=True)


def train(net, items, epochs, lr, bs=32, log=print):
    import torch
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(items) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.15)
    dl = torch.utils.data.DataLoader(DS(items), batch_size=bs, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    for ep in range(epochs):
        net.train(); t0 = time.time(); tot = n = 0
        for x, idx in dl:
            loss = ctc(net(x), idx.tolist(), items).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); n += len(idx)
        log("epoch %d/%d loss %.3f  %.0fs" % (ep + 1, epochs, tot / max(n, 1), time.time() - t0))
        export(net, OUT + ".partial")
    return net


def losses(net, items):
    import torch
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(items), 128):
            ch = list(range(s, min(len(items), s + 128)))
            x = torch.stack([torch.from_numpy(PR.prep(items[i]["crop"]))[None] for i in ch])
            out += ctc(net(x), ch, items).tolist()
    return np.array(out)


def export(net, path):
    TD.export(net, path, {"kind": "poid"})


def evaluate(items, model=OUT, log=print):
    import poid_reader as PR
    PR.MODEL_PATH = model; PR._M = None
    rows = []
    for it in items:
        r = PR.read_po_id(it["crop"])
        rows.append((it["docket"], r["text"], r["conf"]))
    for g in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        ans = [(d, t) for d, t, c in rows if c >= g and t]
        ex = sum(d == t for d, t in ans)
        l8 = sum(len(t) >= 8 and d[-8:] == t[-8:] for d, t in ans)
        log(f"gate {g:.2f}: answered {len(ans)}/{len(rows)} ({len(ans) / max(1, len(rows)):.0%}) | exact {ex / max(1, len(ans)):.1%} | last-8 {l8 / max(1, len(ans)):.1%}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--round2", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    tr, ho = load()
    print(f"train {len(tr)} | hold-out {len(ho)} (never trained on)", flush=True)
    if a.eval:
        evaluate(ho)
        return
    net = TD.make_model()
    if a.round2 and os.path.exists(OUT):
        TD.load_npz(net, OUT)
        L = losses(net, tr)
        keep = L <= np.percentile(L, 85)
        tr = [x for x, k in zip(tr, keep) if k]
        print(f"round 2: dropped the {int((~keep).sum())} worst-agreeing training forms", flush=True)
        train(net, tr, max(4, a.epochs // 2), a.lr / 3)
    else:
        train(net, tr, a.epochs, a.lr)
    export(net, OUT)
    print("saved", OUT, flush=True)
    evaluate(ho)


if __name__ == "__main__":
    main()
