#!/usr/bin/env python
"""Train the handwritten-date CRNN (CNN + BiGRU + CTC) -> models/dates_crnn.npz  (numpy inference in date_reader.py).

  python tools/extract_date_crops.py            # once: data/date_crops.pkl (4 date boxes per form)
  python train_dates.py [--epochs 30] [--round2]  # train; --round2 drops high-loss (label/form disagree) samples and fine-tunes
  python train_dates.py --eval                  # measure on held-out forms (needs the saved model)

Weak labels: the app's loss date / farmer-intimation date (fields 12/13) for every form in data/forms, the disputed-cases
Excel (incidence, intimation, sowing). The way a date was WRITTEN (1/2-digit day & month, 2/4-digit year) is unknown, so the CTC
loss is the marginal likelihood over the 8 writing variants (dd|d x mm|m x yy|yyyy). Never trained on: the 150 hand-labelled
dockets (data/eval/dates.csv), a 10% farmer-grouped hold-out of data/forms, and a farmer-grouped half of the disputed forms.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import pickle
import random
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import date_reader as DR      # noqa: E402

XL_L1 = "/root/.claude/uploads/04c008d2-2f54-5203-acea-b5dca2e7e658/6828e67f-Level_1_GEO_Tagged_QC_Done.xlsx"
XL_DISP = "/root/.claude/uploads/04c008d2-2f54-5203-acea-b5dca2e7e658/1e9a71c6-Disputed_Cases-396.xlsx"
FIELDS = ["sow", "loss", "intim", "insp"]


# ------------------------------------------------------------------ data
def _h(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def _ist_date(v):
    import pandas as pd
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        t = pd.Timestamp(v)
        if t.tzinfo is not None:
            t = t.tz_convert("Asia/Kolkata")
        return t.date()
    except Exception:      # noqa: BLE001
        return None


def load_labels():
    """-> dict[(src, docket)] = dict(group=str, labels={field: date|None}); src 'main'|'disp'"""
    import pandas as pd
    out = {}
    df = pd.read_excel(XL_L1, usecols=["Docket_ID", "Farmer_Name", "Date_Of_Incidence", "Farmer_Intimation_Date"])
    for r in df.itertuples(index=False):
        dk = str(r.Docket_ID).strip().zfill(18)
        out[("main", dk)] = dict(group="m:" + str(r.Farmer_Name), labels={
            "loss": _ist_date(r.Date_Of_Incidence), "intim": _ist_date(r.Farmer_Intimation_Date)})
    dd = pd.read_excel(XL_DISP)
    for r in dd.itertuples(index=False):
        d = dd.columns.get_loc
    for _, r in dd.iterrows():
        dk = str(int(r["docketID"])).zfill(18)
        out[("disp", dk)] = dict(group="d:%s|%s" % (r["farmerName"], str(r["dateOfIncidence"])[:10]), labels={
            "sow": _ist_date(r["sowingDate"]), "loss": _ist_date(r["dateOfIncidence"]),
            "intim": _ist_date(r["farmerIntimationDate"])}, completed=_ist_date(r["surveyCompletedOn"]))
    return out


def hand_labels():
    ev = {}
    p = os.path.join(ROOT, "data", "eval", "dates.csv")
    for r in csv.DictReader(open(p)):
        lab = {}
        for f, k in zip(FIELDS, ("sow", "loss", "intim", "insp")):
            v = (r[k] or "").strip()
            lab[f] = datetime.date(int(v[4:]), int(v[2:4]), int(v[:2])) if len(v) == 8 and v.isdigit() else None
        ev[r["docket"].zfill(18)] = lab
    return ev


def split_role(src, dk, group, hand):
    """'eval_hand' | 'eval_main' | 'eval_disp' | 'train'"""
    if dk in hand:
        return "eval_hand"
    if src == "main":
        return "eval_main" if _h(group) % 10 == 0 else "train"
    return "eval_disp" if _h(group) % 2 == 0 else "train"


def build(crops_pkl, round_labels=None):
    crops = pickle.load(open(crops_pkl, "rb"))
    labs = load_labels()
    hand = hand_labels()
    items = []         # dict(src, dk, field, crop, date, role)
    for (src, dk), cs in crops.items():
        if cs is None:
            continue
        info = labs.get((src, dk))
        group = info["group"] if info else "x:" + dk
        role = split_role(src, dk, group, hand)
        for fi, f in enumerate(FIELDS):
            c = cs[fi]
            if c is None:
                continue
            if role == "eval_hand":
                lab = hand[dk].get(f) if dk in hand else None
            else:
                lab = info["labels"].get(f) if info else None
            if role == "eval_main" and f != "loss":
                lab = None          # the app intimation date differs from the form's in ~80% of forms: not a usable truth
            it = dict(src=src, dk=dk, field=f, crop=c, date=lab, role=role)
            if role == "train" and info:
                L = info["labels"]
                D = datetime.timedelta
                if f == "intim" and L.get("intim"):          # form date = app intimation date + 0..3 days (hand labels: 19/47/16/9 %)
                    it["dates"] = [L["intim"] + D(days=k) for k in range(-1, 6)]
                elif f == "insp" and L.get("intim"):         # committee inspection happens after the intimation, within ~3 weeks
                    it["dates"] = [L["intim"] + D(days=k) for k in range(0, 24)]
                elif f in ("loss", "sow") and lab:
                    it["dates"] = [lab]
            items.append(it)
    return items


# ------------------------------------------------------------------ targets
_VC = {}


def variants_multi(dates):
    out = set()
    for d in dates:
        if d not in _VC:
            _VC[d] = [tuple(v) for v in variants(d)]
        out.update(_VC[d])
    return [list(v) for v in sorted(out)]


def variants(d):
    """date -> list of distinct int-lists (CTC classes) for the writing variants; digits 1..10, separator 11"""
    out = set()
    for dd in {"%02d" % d.day, str(d.day)}:
        for mm in {"%02d" % d.month, str(d.month)}:
            for yy in ("26", "2026"):
                s = dd + "-" + mm + "-" + yy
                out.add(tuple(DR.SEP if ch == "-" else int(ch) + 1 for ch in s))
    return [list(v) for v in sorted(out)]


# ------------------------------------------------------------------ augmentation
def augment(g, rng, trim=True):
    """grey uint8 crop -> grey uint8 crop (random geometry/photometry)"""
    h, w = g.shape
    # random margin trim / pad
    t = [int(rng.uniform(-0.08, 0.10 if trim else 0.0) * s) for s in (h, h, w, w)]
    y0, y1, x0, x1 = max(0, t[0]), min(h, h - t[1]), max(0, t[2]), min(w, w - t[3])
    if y1 - y0 > 8 and x1 - x0 > 8:
        g = g[y0:y1, x0:x1]
    h, w = g.shape
    med = int(np.median(g))
    # affine: rotation, shear, scale
    ang = rng.normal(0, 2.0)
    sh = rng.normal(0, 0.12) if rng.random() < 0.9 else rng.uniform(-0.35, 0.35)
    sx, sy = rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    M = np.array([[sx, sh, 0], [0, sy, 0]], np.float32)
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], np.float32)
    A = R @ np.vstack([M, [0, 0, 1]])
    c = np.array([w / 2, h / 2, 1.0])
    A[:2, 2] = (c - A @ c)[:2] + [rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.06, 0.06) * h]
    pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if rng.random() < 0.4:                       # perspective jitter
        j = rng.normal(0, 0.03, (4, 2)) * [w, h]
        dst = (pts @ A[:2, :2].T + A[:2, 2]) + j
        P = cv2.getPerspectiveTransform(pts, dst.astype(np.float32))
        g = cv2.warpPerspective(g, P, (w, h), borderMode=cv2.BORDER_REPLICATE)
    else:
        g = cv2.warpAffine(g, A[:2], (w, h), borderMode=cv2.BORDER_REPLICATE)
    g = g.astype(np.float32)
    # stroke width
    r = rng.random()
    if r < 0.2:
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    elif r < 0.3:
        g = cv2.dilate(g, np.ones((2, 2), np.uint8))
    # contrast / gamma / brightness
    lo = float(np.percentile(g, 1)); hi = float(np.percentile(g, 99))
    g = (g - lo) / max(hi - lo, 1)
    g = np.clip(g, 0, 1) ** rng.uniform(0.6, 1.8)
    ctr = rng.uniform(0.35, 1.0)
    g = (1 - ctr) * rng.uniform(0.4, 0.9) + ctr * g
    g = g * rng.uniform(150, 255)
    # remnant lines
    for _ in range(rng.integers(0, 3)):
        v = rng.uniform(20, 120)
        th = int(rng.integers(1, 4))
        if rng.random() < 0.6:
            y = int(rng.choice([rng.integers(0, 6), h - 1 - rng.integers(0, 6)]) if rng.random() < 0.8 else rng.integers(0, h))
            cv2.line(g, (0, y), (w, y + int(rng.integers(-3, 4))), v, th)
        else:
            x = int(rng.choice([rng.integers(0, 6), w - 1 - rng.integers(0, 6)]))
            cv2.line(g, (x, 0), (x + int(rng.integers(-3, 4)), h), v, th)
    # blur / downscale / noise
    if rng.random() < 0.5:
        k = rng.uniform(0.4, 1.8)
        g = cv2.GaussianBlur(g, (0, 0), k)
    if rng.random() < 0.3:
        f = rng.uniform(0.4, 0.8)
        g = cv2.resize(cv2.resize(g, (max(8, int(w * f)), max(8, int(h * f)))), (w, h))
    g = g + rng.normal(0, rng.uniform(0, 12), g.shape)
    if rng.random() < 0.15:                      # partial occlusion by a stain
        x = int(rng.integers(0, w)); ww = int(rng.integers(4, max(5, w // 8)))
        g[:, x:x + ww] = g[:, x:x + ww] * rng.uniform(0.6, 1.0) + rng.uniform(0, 40)
    return np.clip(g, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ synthetic dates (MNIST digits): breaks the language prior
_MN = None


def _mnist():
    global _MN
    if _MN is None:
        import gzip
        base = os.path.join(ROOT, "data", "mnist")
        with gzip.open(os.path.join(base, "train-images-idx3-ubyte.gz")) as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
        with gzip.open(os.path.join(base, "train-labels-idx1-ubyte.gz")) as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8)
        _MN = ([X[y == k] for k in range(10)])
    return _MN


def synth_item(rng):
    """-> dict(date, tgt (exact CTC target), style, synthetic=True)"""
    mo = int(rng.integers(3, 13)) if rng.random() < 0.8 else int(rng.integers(1, 13))
    d = datetime.date(2026, mo, int(rng.integers(1, 29)))
    dd = "%02d" % d.day if rng.random() < 0.55 else str(d.day)
    mm = "%02d" % d.month if rng.random() < 0.55 else str(d.month)
    yy = "26" if rng.random() < 0.55 else "2026"
    sep = ["-", "/", "."][int(rng.integers(0, 3))] if rng.random() < 0.9 else "-"
    txt = dd + sep + mm + sep + yy
    tgt = [DR.SEP if ch in "-/." else int(ch) + 1 for ch in txt]
    return dict(date=d, tgt=tgt, text=txt, synthetic=True, field="synth", role="train", dk="synth", src="synth")


def render_synth(it, rng):
    M = _mnist()
    H = 96
    ink = rng.uniform(10, 90)
    glyphs = []
    for ch in it["text"]:
        if ch.isdigit():
            g = M[int(ch)][rng.integers(0, len(M[int(ch)]))]
            ys, xs = np.nonzero(g > 60)
            g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
            hh = int(H * rng.uniform(0.45, 0.8))
            ww = max(3, int(g.shape[1] * hh / g.shape[0] * rng.uniform(0.7, 1.2)))
            g = cv2.resize(g, (ww, hh), interpolation=cv2.INTER_AREA)
            if rng.random() < 0.7:                                  # thicken / thin strokes to match pen width
                g = cv2.dilate(g, np.ones((2, 2), np.uint8)) if rng.random() < 0.6 else g
            glyphs.append((g, rng.uniform(-0.1, 0.45)))
        else:
            hh = int(H * rng.uniform(0.4, 0.8))
            g = np.zeros((hh, max(4, hh // 3)), np.float32)
            if ch == "-":
                g = np.zeros((6, int(H * rng.uniform(0.25, 0.5))), np.float32); g[2:4] = 1
            elif ch == "/":
                cv2.line(g, (g.shape[1] - 1, 0), (0, hh - 1), 1.0, 2)
            else:
                g = np.zeros((6, 6), np.float32); g[1:5, 1:5] = 1
            glyphs.append((g, 0.0))
    gap = [int(rng.integers(2, 22)) for _ in glyphs]
    W = sum(g.shape[1] for g, _ in glyphs) + sum(gap) + 40
    cv = np.zeros((H, W), np.float32)
    x = int(rng.integers(5, 25))
    base = rng.uniform(0.0, 0.2)
    for (g, sl), gp in zip(glyphs, gap):
        h, w = g.shape
        if h < 8:           # '-' / '.' sit mid-height or low
            y0 = int(H * (0.55 if g.shape[0] == 6 and g.shape[1] > 8 else 0.7))
        else:
            y0 = int(H * (1 - base) - h) - int(rng.integers(-4, 5))
        y0 = int(np.clip(y0, 0, H - h)); w = min(w, W - x)
        if w > 0:
            cv[y0:y0 + h, x:x + w] = np.maximum(cv[y0:y0 + h, x:x + w], g[:, :w])
        x += w + gp
    sh = rng.uniform(-0.2, 0.2)
    cv = cv2.warpAffine(cv, np.array([[1, sh, -sh * H / 2], [0, 1, 0]], np.float32), (W, H))
    paper = rng.uniform(150, 245)
    img = paper - cv * (paper - ink) + rng.normal(0, 4, cv.shape)
    for _ in range(int(rng.integers(0, 3))):                      # ruled lines
        y = int(rng.choice([2, H - 3])); cv2.line(img, (0, y), (W, y), paper * rng.uniform(0.4, 0.8), 2)
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ model
def make_model():
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()

            def cb(i, o):
                return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]
            self.cnn = nn.Sequential(*cb(1, 16), nn.MaxPool2d(2), *cb(16, 32), nn.MaxPool2d(2), *cb(32, 64),
                                     *cb(64, 64), nn.MaxPool2d((2, 1)), *cb(64, 96), nn.MaxPool2d((2, 1)))
            self.fc = nn.Linear(96 * 3, 128)
            self.drop = nn.Dropout(0.2)
            self.gru = nn.GRU(128, 96, batch_first=True, bidirectional=True)
            self.out = nn.Linear(192, 12)

        def forward(self, x):                      # x (B,1,48,256) -> logits (B,T,12)
            h = self.cnn(x)
            B, C, H, T = h.shape
            h = h.permute(0, 3, 1, 2).reshape(B, T, C * H)
            h = self.drop(torch.relu(self.fc(h)))
            h, _ = self.gru(h)
            return self.out(self.drop(h))
    return Net()


def export(net, path, meta):
    import torch
    net.eval()
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    out = {}
    convs = [(0, "c1"), (3, "c2"), (6, "c3"), (9, "c4"), (13, "c5")]
    idx = [k for k in range(len(net.cnn)) if isinstance(net.cnn[k], torch.nn.Conv2d)]
    for (k, name), k2 in zip(convs, idx):
        w = sd["cnn.%d.weight" % k2]
        bnk = k2 + 1
        g, b, m, v = (sd["cnn.%d.%s" % (bnk, s)] for s in ("weight", "bias", "running_mean", "running_var"))
        s = g / np.sqrt(v + 1e-5)
        out[name + "w"] = (w * s[:, None, None, None]).astype(np.float32)
        out[name + "b"] = (b - m * s).astype(np.float32)
    out["fw"], out["fb"] = sd["fc.weight"], sd["fc.bias"]
    for suf, nm in (("", "g0"), ("_reverse", "g0r")):
        for a in ("wih", "whh", "bih", "bhh"):
            out["%s_%s" % (nm, a)] = sd["gru.%s_l0%s" % ({"wih": "weight_ih", "whh": "weight_hh", "bih": "bias_ih", "bhh": "bias_hh"}[a], suf)]
    out["ow"], out["ob"] = sd["out.weight"], sd["out.bias"]
    for k, v in meta.items():
        out[k] = np.array(v)
    np.savez_compressed(path, **out)


def load_npz(net, path):
    """initialise the torch net from an exported npz (BN folded -> identity BN with the folded bias)"""
    import torch
    z = np.load(path)
    idx = [k for k in range(len(net.cnn)) if isinstance(net.cnn[k], torch.nn.Conv2d)]
    sd = net.state_dict()
    T = lambda a: torch.from_numpy(np.asarray(a, np.float32))
    for name, k2 in zip(("c1", "c2", "c3", "c4", "c5"), idx):
        sd["cnn.%d.weight" % k2].copy_(T(z[name + "w"]))
        o = z[name + "b"].shape[0]
        sd["cnn.%d.weight" % (k2 + 1)].copy_(torch.ones(o)); sd["cnn.%d.bias" % (k2 + 1)].copy_(T(z[name + "b"]))
        sd["cnn.%d.running_mean" % (k2 + 1)].copy_(torch.zeros(o)); sd["cnn.%d.running_var" % (k2 + 1)].copy_(torch.full((o,), 1 - 1e-5))
    sd["fc.weight"].copy_(T(z["fw"])); sd["fc.bias"].copy_(T(z["fb"]))
    for suf, nm in (("", "g0"), ("_reverse", "g0r")):
        for a, tn in (("wih", "weight_ih"), ("whh", "weight_hh"), ("bih", "bias_ih"), ("bhh", "bias_hh")):
            sd["gru.%s_l0%s" % (tn, suf)].copy_(T(z["%s_%s" % (nm, a)]))
    sd["out.weight"].copy_(T(z["ow"])); sd["out.bias"].copy_(T(z["ob"]))
    return net


def relabel(items, log=print, gate=0.7, max_days=10):
    """self-training: replace the (noisy) app label by the model's own reading when the reading is a valid, confident date
    within +-max_days of the app label (loss/intim/sow) or, where no label exists (insp), whenever it is confident."""
    changed = kept = fresh = dropped = 0
    out = []
    for it in items:
        r = DR.read_date(it["crop"], it["field"])
        rd = datetime.datetime.strptime(r["date"], "%d%m%Y").date() if r["date"] else None
        lab = it["date"]
        if rd is not None and r["conf"] >= gate:
            if lab is None:
                it = dict(it, date=rd); fresh += 1
            elif rd == lab:
                kept += 1
            elif abs((rd - lab).days) <= max_days:
                it = dict(it, date=rd); changed += 1
            else:
                dropped += 1; continue
        elif lab is None:
            continue
        else:
            dropped += 1; continue          # unreadable / uncertain / blank box: unknown whether label is right
        out.append(it)
    log("relabel: kept %d changed %d fresh %d dropped %d" % (kept, changed, fresh, dropped))
    return out


# ------------------------------------------------------------------ training
class DS:
    def __init__(self, items, aug=True):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        rng = np.random.default_rng((i * 7919 + int(time.time() * 1000)) % (2 ** 32))
        if it.get("synthetic"):
            g = augment(render_synth(it, rng), rng, trim=False)
        else:
            g = augment(it["crop"], rng) if self.aug else it["crop"]
        import torch
        return torch.from_numpy(DR.prep(g))[None], i


def ctc_marginal(net_out, idxs, items, drop_set=None):
    """net_out logits (B,T,12); per-sample -log sum_v exp(-nll_v) over writing variants -> tensor (B,)"""
    import torch
    import torch.nn.functional as F
    lp = F.log_softmax(net_out, -1).permute(1, 0, 2)          # T,B,C
    T = lp.shape[0]
    tgts, lens, owner = [], [], []
    for bi, i in enumerate(idxs):
        for v in ([items[i]["tgt"]] if items[i].get("synthetic") else variants_multi(items[i]["dates"])):
            tgts += v; lens.append(len(v)); owner.append(bi)
    owner_t = torch.tensor(owner)
    lpv = lp[:, owner_t]
    nll = F.ctc_loss(lpv, torch.tensor(tgts), torch.full((len(owner),), T, dtype=torch.long), torch.tensor(lens),
                     blank=0, reduction="none", zero_infinity=False)
    nll = torch.nan_to_num(nll, posinf=1e3)
    B = len(idxs)
    M = torch.full((B, len(owner)), float("inf"))
    M[owner_t, torch.arange(len(owner))] = nll
    return -torch.logsumexp(-M, 1)


def run_epochs(net, items, epochs, lr, bs=16, log=print):
    import torch
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(items) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    dl = torch.utils.data.DataLoader(DS(items), batch_size=bs, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    for ep in range(epochs):
        net.train(); t0 = time.time(); tot = 0; n = 0
        for x, idx in dl:
            loss_s = ctc_marginal(net(x), idx.tolist(), items)
            loss = loss_s.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); sched.step()
            tot += float(loss.detach()) * len(idx); n += len(idx)
        log("epoch %d/%d loss %.3f  %.0fs" % (ep + 1, epochs, tot / n, time.time() - t0))
    return net


def sample_losses(net, items):
    import torch
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(items), 128):
            ch = list(range(s, min(len(items), s + 128)))
            x = torch.stack([torch.from_numpy(DR.prep(items[i]["crop"]))[None] for i in ch])
            out += ctc_marginal(net(x), ch, items).tolist()
    return np.array(out)


# ------------------------------------------------------------------ evaluation
def read_items(items, field_window=True):
    res = []
    for it in items:
        r = DR.read_date(it["crop"], it["field"] if field_window else None)
        res.append(r)
    return res


def style_of(text):
    """writing style of a decoded text: (sep kind, 2- vs 4-digit year, 1-digit day/month present)"""
    import re
    kind = "none"
    for ch, nm in (("/", "slash"), ("-", "sep")):
        pass
    parts = [p for p in text.split("-") if p]
    ylen = len(parts[2]) if len(parts) >= 3 else None
    ys = "yy" if ylen == 2 else ("yyyy" if ylen == 4 else "?")
    short = "d/m short" if len(parts) >= 3 and (len(parts[0]) == 1 or len(parts[1]) == 1) else "dd/mm"
    return ys, short


def evaluate(items, gates=(0.0, 0.5, 0.7, 0.8, 0.9, 0.95), log=print):
    res = read_items(items)
    rows = []
    for field in FIELDS:
        for role in ("eval_main", "eval_disp", "eval_hand"):
            sel = [(it, r) for it, r in zip(items, res) if it["field"] == field and it["role"] == role and it["date"] is not None]
            if not sel:
                continue
            truth = [it["date"].strftime("%d%m%Y") for it, _ in sel]
            line = "%-5s %-9s n=%4d |" % (field, role, len(sel))
            for gte in gates:
                ans = [(t, r) for t, (_, r) in zip(truth, sel) if r["date"] and r["conf"] >= gte]
                ok = sum(t == r["date"] for t, r in ans)
                line += " g%.2f cov %3.0f%% prec %5.1f%% |" % (gte, 100 * len(ans) / len(sel), 100 * ok / max(1, len(ans)))
            log(line)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default=os.path.join(ROOT, "data", "date_crops.pkl"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--synth", type=int, default=0, help="number of synthetic MNIST-digit dates per epoch set")
    ap.add_argument("--init", default=None, help="warm start from this npz")
    ap.add_argument("--relabel", action="store_true", help="start from --out, self-label (see relabel()), fine-tune")
    ap.add_argument("--round2", action="store_true", help="drop the worst 12%% (label/form disagree) and fine-tune")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--out", default=DR.MODEL_PATH)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    random.seed(0); torch.manual_seed(0); np.random.seed(0)
    items = build(a.crops)
    if a.eval:
        evaluate([i for i in items if i["role"] != "train"])
        return
    train = [i for i in items if i["role"] == "train" and i.get("dates")]
    print("train samples", len(train), "by field", {f: sum(i["field"] == f for i in train) for f in FIELDS})
    net = make_model()
    if a.synth:
        r_ = np.random.default_rng(1)
        train = train + [synth_item(r_) for _ in range(a.synth)]
    if a.init:
        load_npz(net, a.init)
    if a.relabel:
        load_npz(net, a.out)
        train = relabel([i for i in items if i["role"] == "train"])
        print("relabelled train samples", len(train), {f: sum(i["field"] == f for i in train) for f in FIELDS})
    net = run_epochs(net, train, a.epochs, a.lr)
    if a.round2:
        L = sample_losses(net, train)
        thr = np.percentile(L, 88)
        keep = [it for it, l in zip(train, L) if l <= thr]
        print("round 2: keep", len(keep), "of", len(train), "loss thr %.2f" % thr)
        net = run_epochs(net, keep, max(8, a.epochs // 2), a.lr / 3)
    export(net, a.out, dict(conf_gate=0.8))
    DR._M = None
    evaluate([i for i in items if i["role"] != "train"])


if __name__ == "__main__":
    main()
