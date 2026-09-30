"""Offline handwritten-number reader for the PMFBY Proforma-3 forms (see models/README.md).

Public API (stable):
  read_number(img, *, max_digits=3, allow_decimal=True) -> dict(value, text, conf, n_components)
  read_digit_string(img, expected_len=None)             -> dict(text, conf, n_components)
"""
import os
import re

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "digits_cnn.npz")
# classes: 0-9 digits, 10 '.', 11 '%'/slash/other trailing mark, 12 junk
CLASSES = list("0123456789") + [".", "%", "#"]
NCLS = len(CLASSES)
CELL_H = 64


# ----------------------------------------------------------------- preprocessing
def to_gray(img):
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"))
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return a.astype(np.uint8)


def ink_mask(g, h=CELL_H):
    """gray cell -> (binary ink mask at height h, scale). Ruled lines removed, background flattened."""
    g = to_gray(g)
    s = h / g.shape[0]
    g = cv2.resize(g, (max(8, int(g.shape[1] * s)), h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    bg = cv2.medianBlur(cv2.dilate(g, np.ones((1, 1), np.uint8)), 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    flat = cv2.divide(g, np.maximum(bg, 1), scale=255)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    # ink = clearly darker than local background
    thr = min(215, max(120, int(np.percentile(flat, 1) + 0.5 * (255 - np.percentile(flat, 1)) * 0.55)))
    _, b = cv2.threshold(flat, thr, 255, cv2.THRESH_BINARY_INV)
    H, W = b.shape
    hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, int(W * 0.45)), 1)))
    vl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(H * 0.75)))))
    lines = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
    b = cv2.bitwise_and(b, cv2.bitwise_not(lines))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return b, s


def components(b, min_area=18):
    """binary mask -> list of dicts (x0,y0,x1,y1,mask) sorted left-to-right; overlapping-in-x parts merged."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    H, W = b.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < min_area:
            continue
        if x <= 1 and w <= 4:
            continue  # cell border remnant
        boxes.append([x, y, x + w, y + h, [i]])
    boxes.sort(key=lambda t: t[0])
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                ov = min(a[2], c[2]) - max(a[0], c[0])
                wmin = min(a[2] - a[0], c[2] - c[0])
                # a '%' or an 'i'-like mark: parts stacked with big x overlap; merge only when one is small
                smallh = min(a[3] - a[1], c[3] - c[1])
                if ov > 0.6 * wmin and smallh < 0.55 * H and wmin > 0:
                    boxes[i] = [min(a[0], c[0]), min(a[1], c[1]), max(a[2], c[2]), max(a[3], c[3]), a[4] + c[4]]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    out = []
    for x0, y0, x1, y1, ids in sorted(boxes, key=lambda t: t[0]):
        m = np.isin(lab[y0:y1, x0:x1], ids).astype(np.uint8) * 255
        out.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, mask=m, area=int((m > 0).sum())))
    return out


def norm28(comp, size=28, box=20):
    """component mask -> MNIST-like 28x28 float (0..1), aspect-preserving, centred by centre of mass."""
    m = comp["mask"]
    h, w = m.shape
    sc = box / max(h, w)
    nh, nw = max(1, round(h * sc)), max(1, round(w * sc))
    r = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_AREA)
    r = cv2.GaussianBlur(r, (3, 3), 0) if min(nh, nw) > 3 else r
    canvas = np.zeros((size, size), np.uint8)
    ys, xs = np.nonzero(r)
    if len(ys) == 0:
        return canvas.astype(np.float32) / 255
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    oy, ox = size // 2 - cy, size // 2 - cx
    oy = min(max(oy, 0), size - nh)
    ox = min(max(ox, 0), size - nw)
    canvas[oy:oy + nh, ox:ox + nw] = r
    return canvas.astype(np.float32) / 255


def comp_feats(comp, H):
    """geometry features used alongside the CNN: height/H, width/height, centre y / H, bottom/H."""
    w, h = comp["x1"] - comp["x0"], comp["y1"] - comp["y0"]
    return np.array([h / H, w / max(h, 1), (comp["y0"] + comp["y1"]) / 2 / H, comp["y1"] / H], np.float32)


# ----------------------------------------------------------------- numpy CNN inference
_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        z = np.load(MODEL_PATH)
        _MODEL = {k: z[k].astype(np.float32) for k in z.files}
    return _MODEL


def _conv3(x, w, b):
    """x (N,C,H,W) same-padded 3x3 conv; w (O,C,3,3)."""
    n, c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    win = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(2, 3))  # N,C,H,W,3,3
    return np.einsum("nchwij,ocij->nohw", win, w, optimize=True) + b[None, :, None, None]


def _pool(x):
    n, c, h, w = x.shape
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def predict_proba(x28):
    """x28: (N,28,28) float 0..1 -> (N, NCLS) probabilities."""
    m = _load()
    x = x28[:, None].astype(np.float32)
    x = _pool(np.maximum(_conv3(x, m["c1w"], m["c1b"]), 0))
    x = _pool(np.maximum(_conv3(x, m["c2w"], m["c2b"]), 0))
    x = x.reshape(len(x), -1)
    x = np.maximum(x @ m["f1w"].T + m["f1b"], 0)
    z = x @ m["f2w"].T + m["f2b"]
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def classify_components(comps):
    if not comps:
        return np.zeros((0, NCLS), np.float32)
    return predict_proba(np.stack([norm28(c) for c in comps]))


# ----------------------------------------------------------------- assembly
def _analyse(img, h=CELL_H):
    b, s = ink_mask(img, h)
    H, W = b.shape
    comps = components(b)
    keep = []
    for c in comps:
        ch, cw = c["y1"] - c["y0"], c["x1"] - c["x0"]
        cy = (c["y0"] + c["y1"]) / 2 / H
        if ch < 0.12 * H and cw < 0.14 * H:  # speck / dot candidate
            c["dot"] = cy > 0.55 and c["area"] >= 14
            if not c["dot"]:
                continue
        else:
            c["dot"] = False
            if ch < 0.28 * H and cw >= 0.5 * H:  # flat dash / leftover rule piece
                continue
        keep.append(c)
    return keep, H, W


def _seq(comps, probs, H):
    """-> list of (char, prob) in reading order, junk dropped; also trailing-mark count."""
    out = []
    for c, p in zip(comps, probs):
        if c["dot"]:
            out.append((".", 0.9, c))
            continue
        k = int(p.argmax())
        ch = CLASSES[k]
        out.append((ch, float(p[k]), c))
    return out


def read_digit_string(img, expected_len=None):
    comps, H, W = _analyse(img)
    comps = [c for c in comps if not c["dot"]]
    if not comps:
        return dict(text="", conf=0.0, n_components=0)
    probs = classify_components(comps)
    dig = probs[:, :10]
    txt = "".join(str(int(r.argmax())) for r in dig)
    conf = float(np.mean(dig.max(1) / np.maximum(dig.sum(1), 1e-9)))
    return dict(text=txt, conf=conf, n_components=len(comps))


def read_number(img, *, max_digits=3, allow_decimal=True, min_conf=0.5):
    comps, H, W = _analyse(img)
    if not comps:
        return dict(value=None, text="", conf=1.0, n_components=0)
    probs = classify_components(comps)
    seq = _seq(comps, probs, H)
    chars, confs = [], []
    seen_mark = False
    for ch, p, c in seq:
        if ch in ("%", "#"):
            if ch == "%":
                seen_mark = True
            continue
        if seen_mark:  # digits after a mark -> ambiguous
            confs.append(0.0)
        if ch == "." and (not allow_decimal or "." in chars):
            continue
        chars.append(ch)
        confs.append(p)
    # strip leading/trailing dots
    text = "".join(chars).strip(".") if chars else ""
    conf = float(min(confs)) if confs else 0.0
    n = len(comps)
    if not text or not any(ch.isdigit() for ch in text):
        return dict(value=None, text=text, conf=min(conf, 0.3), n_components=n)
    try:
        val = float(text)
    except ValueError:
        return dict(value=None, text=text, conf=min(conf, 0.3), n_components=n)
    nd = sum(ch.isdigit() for ch in text)
    if nd > max_digits + (2 if "." in text else 0) or val > 100:
        return dict(value=None, text=text, conf=min(conf, 0.4), n_components=n)
    if val == int(val):
        val = float(int(val))
    return dict(value=val if conf >= min_conf else None, text=text, conf=conf, n_components=n)
