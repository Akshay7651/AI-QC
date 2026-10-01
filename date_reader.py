"""Offline handwritten-DATE reader for the 4 date boxes of the PMFBY Proforma-3 header grid.

    read_date(crop_img, field=None) -> dict(date='DDMMYYYY'|None, text=str, conf=float, ...)

A small CNN + BiGRU trained with CTC (train_dates.py) reads the whole box as a character SEQUENCE (digits 0-9 plus one
separator class that stands for '-', '/', '.'), so no digit segmentation is needed. Inference is pure numpy
(models/dates_crnn.npz). `text` is the raw decoded string (e.g. '10-06-26'); `date` is the parsed calendar date
(always year 2026 here) or None when it cannot be parsed to ONE plausible date.

Handles 1/2-digit day and month, 2/4-digit year, any separator (or none when the length is unambiguous).
"""
import datetime
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "dates_crnn.npz")
IH, IW = 48, 256
SEP = 11                      # class ids: 0 blank, 1..10 digits '0'..'9', 11 separator
YEAR = 2026
# plausibility windows per field (inclusive): sowing before the loss; the other three after it
WINDOWS = {
    "sow": (datetime.date(2026, 3, 1), datetime.date(2026, 9, 30)),
    "loss": (datetime.date(2026, 6, 1), datetime.date(2026, 12, 31)),
    "intim": (datetime.date(2026, 6, 1), datetime.date(2026, 12, 31)),
    "insp": (datetime.date(2026, 6, 1), datetime.date(2026, 12, 31)),
}
DEFAULT_WINDOW = (datetime.date(2026, 3, 1), datetime.date(2026, 12, 31))
CONF_GATE = 0.80              # overwritten from the model file's metadata when present


# ------------------------------------------------------------------ preprocessing (shared with train_dates.py)
def to_gray(img):
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            return np.asarray(img.convert("L"))
    except Exception:      # noqa: BLE001
        pass
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return a.astype(np.uint8)


def prep(g):
    """grey crop (any size) -> float32 (IH, IW): background-flattened ink map in [0,1] (1 = ink)"""
    g = cv2.resize(g, (IW, IH), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    x = 1.0 - g / np.maximum(bg, 1.0)
    x = np.clip(x, 0, None)
    x = x / max(float(np.percentile(x, 99)), 0.12)
    return np.clip(x, 0, 1).astype(np.float32)


# ------------------------------------------------------------------ numpy network
_M = None


def _load():
    global _M, CONF_GATE
    if _M is None:
        if not os.path.exists(MODEL_PATH):
            return None
        z = np.load(MODEL_PATH, allow_pickle=False)
        _M = {k: z[k] for k in z.files}
        if "conf_gate" in _M:
            CONF_GATE = float(_M["conf_gate"])
    return _M


def _conv3(x, w, b):
    """x (C,H,W) same-padded 3x3 conv (BN already folded) + ReLU. w (O,C,3,3)"""
    C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))      # C,H,W,3,3
    cols = cols.transpose(1, 2, 0, 3, 4).reshape(H * W, C * 9)
    y = cols @ w.reshape(w.shape[0], -1).T + b
    return np.maximum(y, 0).reshape(H, W, -1).transpose(2, 0, 1)


def _pool(x, ph, pw):
    C, H, W = x.shape
    return x[:, :H // ph * ph, :W // pw * pw].reshape(C, H // ph, ph, W // pw, pw).max(axis=(2, 4))


def _sig(v):
    return 1.0 / (1.0 + np.exp(-v))


def _gru(x, wih, whh, bih, bhh, reverse=False):
    T = x.shape[0]
    Hd = whh.shape[1]
    gi = x @ wih.T + bih
    h = np.zeros(Hd, np.float32)
    out = np.zeros((T, Hd), np.float32)
    for t in (range(T - 1, -1, -1) if reverse else range(T)):
        gh = whh @ h + bhh
        r = _sig(gi[t, :Hd] + gh[:Hd])
        z = _sig(gi[t, Hd:2 * Hd] + gh[Hd:2 * Hd])
        n = np.tanh(gi[t, 2 * Hd:] + r * gh[2 * Hd:])
        h = (1 - z) * n + z * h
        out[t] = h
    return out


def forward(x, M=None):
    """x float32 (IH,IW) -> softmax probabilities (T, 12)"""
    M = M or _load()
    h = x[None]
    h = _pool(_conv3(h, M["c1w"], M["c1b"]), 2, 2)
    h = _pool(_conv3(h, M["c2w"], M["c2b"]), 2, 2)
    h = _conv3(h, M["c3w"], M["c3b"])
    h = _pool(_conv3(h, M["c4w"], M["c4b"]), 2, 1)
    h = _pool(_conv3(h, M["c5w"], M["c5b"]), 2, 1)          # (C, 3, 64)
    C, H, T = h.shape
    seq = h.transpose(2, 0, 1).reshape(T, C * H)
    seq = np.maximum(seq @ M["fw"].T + M["fb"], 0)
    f = _gru(seq, M["g0_wih"], M["g0_whh"], M["g0_bih"], M["g0_bhh"])
    b = _gru(seq, M["g0r_wih"], M["g0r_whh"], M["g0r_bih"], M["g0r_bhh"], reverse=True)
    lg = np.concatenate([f, b], 1) @ M["ow"].T + M["ob"]
    lg = lg - lg.max(1, keepdims=True)
    e = np.exp(lg)
    return e / e.sum(1, keepdims=True)


def ctc_greedy(P):
    """-> (text, per-char confidences) ; separators are written as '-'"""
    k = P.argmax(1)
    pm = P.max(1)
    chars, confs, prev = [], [], 0
    i = 0
    T = len(k)
    while i < T:
        if k[i] != 0 and k[i] != prev:
            j = i
            while j + 1 < T and k[j + 1] == k[i]:
                j += 1
            chars.append("-" if k[i] == SEP else str(k[i] - 1))
            confs.append(float(pm[i:j + 1].max()))
            i = j + 1
            prev = k[i - 1]
            continue
        prev = k[i]
        i += 1
    return "".join(chars), confs


# ------------------------------------------------------------------ parsing
def _fix_year(y):
    """year string -> True when it is a plausible (possibly damaged) '26' / '2026'"""
    return y in ("26", "2026", "202", "026", "206", "20", "6") or (len(y) in (2, 3, 4) and y.endswith("26"))


def candidates(text, window=DEFAULT_WINDOW):
    """decoded string -> sorted list of distinct valid datetime.date inside the window (year forced to 2026)"""
    lo, hi = window
    out = set()

    def add(d, m, y):
        if not d or not m or len(d) > 2 or len(m) > 2 or not _fix_year(y):
            return
        try:
            dt = datetime.date(YEAR, int(m), int(d))
        except ValueError:
            return
        if lo <= dt <= hi:
            out.add(dt)

    parts = [p for p in text.replace(" ", "").split("-")]
    parts = [p for p in parts if p != ""]
    if len(parts) >= 3:
        # normal case; with extra stray groups try every consecutive triple
        for i in range(0, len(parts) - 2):
            add(parts[i], parts[i + 1], parts[i + 2])
    elif len(parts) == 2:
        a, b = parts                               # e.g. '10-0626' / '1006-26' / '10-06' (year missing)
        s = a + b
        if len(a) <= 2 and len(b) in (3, 4, 5, 6):  # dd-mmyy...
            for k in (1, 2):
                if k <= len(b) - 2:
                    add(a, b[:k], b[k:])
        if len(b) in (2, 4) and len(a) in (2, 3, 4):
            for k in (1, 2):
                if len(a) - k in (1, 2):
                    add(a[:k], a[k:], b)
        if len(s) in (6, 8):
            add(s[:2], s[2:4], s[4:])
    elif len(parts) == 1:
        s = parts[0]
        n = len(s)
        if n in (6, 8):
            add(s[:2], s[2:4], s[4:])
        if n in (4, 5, 6, 7):                      # d m yy / dd m yy / d mm yy / dd mm yy
            for dl in (1, 2):
                for ml in (1, 2):
                    if dl + ml + 2 == n:
                        add(s[:dl], s[dl:dl + ml], s[dl + ml:])
                    if dl + ml + 4 == n:
                        add(s[:dl], s[dl:dl + ml], s[dl + ml:])
    return sorted(out)


_BLANK = {"date": None, "text": "", "conf": 0.0, "blank": True, "ambiguous": False, "cands": []}


def read_date(crop_img, field=None):
    """handwritten date box (PIL / numpy, grey or RGB) -> dict(date 'DDMMYYYY'|None, text, conf, blank, ambiguous, cands).

    field in {'sow','loss','intim','insp'} selects the plausibility window. conf = minimum peak probability over the
    decoded characters (low if any character is doubtful); gate it with CONF_GATE (stored in the model file)."""
    M = _load()
    if M is None:
        return dict(_BLANK, blank=False, note="model missing")
    g = to_gray(crop_img)
    if g.size == 0 or min(g.shape) < 6:
        return dict(_BLANK)
    x = prep(g)
    if float((x > 0.5).sum()) < 30:            # (almost) no ink at all: empty box
        return dict(_BLANK)
    P = forward(x, M)
    text, confs = ctc_greedy(P)
    conf = float(min(confs)) if confs else 0.0
    if not any(ch.isdigit() for ch in text):
        return {"date": None, "text": text, "conf": 1.0 - float(P[:, 1:].max()) if not text else 0.0,
                "blank": not text, "ambiguous": False, "cands": []}
    win = WINDOWS.get(field, DEFAULT_WINDOW)
    c = candidates(text, win)
    d = c[0].strftime("%d%m%Y") if len(c) == 1 else None
    return {"date": d, "text": text, "conf": conf, "blank": False, "ambiguous": len(c) > 1,
            "cands": [x.strftime("%d%m%Y") for x in c]}
