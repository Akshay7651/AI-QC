"""Offline reader for the handwritten PO ID (18-digit docket number) on the Haryana Proforma-3.

    read_po_id(grey_or_rgb_crop) -> dict(text=str, conf=float)

Same network as the date reader (CNN + BiGRU + CTC, numpy inference); trained by train_poid.py on the forms' own docket ids.
conf = the lowest per-digit confidence (one unsure digit makes the whole number unsure).
"""
import os

import numpy as np

import date_reader as DR

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "poid_crnn.npz")
CONF_GATE = 0.9          # hold-out (526 forms): 37% answered; 87% equal the docket exactly - most of the rest are forms where the surveyor wrote another number (checked by eye)
_M = None
IH, IW = 48, 512          # twice the date reader's width: 18 handwritten digits need the resolution


def prep(g):
    """grey line crop -> float32 (IH, IW) ink map, same normalisation as date_reader.prep"""
    import cv2
    g = cv2.resize(g, (IW, IH), interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    x = np.clip(1.0 - g / np.maximum(bg, 1.0), 0, None)
    x = x / max(float(np.percentile(x, 99)), 0.12)
    return np.clip(x, 0, 1).astype(np.float32)


def _load():
    global _M
    if _M is None:
        if not os.path.exists(MODEL_PATH):
            return None
        z = np.load(MODEL_PATH, allow_pickle=False)
        _M = {k: z[k] for k in z.files}
    return _M


def available():
    return _load() is not None


def read_po_id(img):
    M = _load()
    if M is None:
        return {"text": "", "conf": 0.0}
    g = DR.to_gray(img)
    P = DR.forward(prep(g), M)
    text, confs = DR.ctc_greedy(P)
    text = text.replace("-", "")
    return {"text": text, "conf": float(min(confs)) if confs else 0.0}


def window_box(lay):
    """search window (hi-res coords) around the layout's PO-ID box: taller than the box, to the right page edge"""
    x0, y0, x1, y1 = lay["boxes"]["po_id"]
    h = y1 - y0
    return (x0 - 0.1 * (x1 - x0), y0 - 0.6 * h, lay["rgb"].shape[1], y1 + 1.0 * h), h


def find_line(rgb, box_h):
    """rgb search window (from window_box) -> grey crop of the handwritten PO-ID line, or None.
    The line is the row band with the most pen ink (blue pens: blue ink; otherwise dark, non-brown ink), weighted towards the
    middle of the window where the layout box says it should be (this keeps the printed header / the next grid row out)."""
    import cv2
    if rgb is None or rgb.size == 0 or rgb.shape[0] > 6 * max(box_h, 1) or rgb.shape[0] < 10:
        return None
    H = rgb.shape[0]
    c = rgb.astype(np.int16)
    blue = c[..., 2] - c[..., 0]
    s = c.sum(2)
    darkish = s < np.percentile(s, 35)
    pen_blue = (blue - np.median(blue) > max(10.0, float(np.percentile(blue, 97) - np.median(blue)) * 0.45)) & darkish
    dark = (s < np.percentile(s, 8)) & ((c[..., 0] - c[..., 2]) < 15)          # dark and not brown/red print
    # the barcode (dense vertical bars) is dark too: blank out columns/rows dominated by tall thin vertical strokes
    import cv2 as _cv
    bars = _cv.morphologyEx(dark.astype(np.uint8), _cv.MORPH_OPEN, _cv.getStructuringElement(_cv.MORPH_RECT, (1, max(5, int(0.45 * box_h)))))
    if bars.any():
        ys, xs = np.nonzero(bars)
        dark[max(0, ys.min() - 3): ys.max() + 4, max(0, xs.min() - 3): xs.max() + 4] = False
        pen_blue[max(0, ys.min() - 3): ys.max() + 4, max(0, xs.min() - 3): xs.max() + 4] = False
    centre = (0.6 * box_h + 0.55 * box_h) / (2.6 * box_h) * H                     # middle of the layout box inside the window
    prior = np.exp(-0.5 * ((np.arange(H) - centre) / (0.7 * box_h)) ** 2)
    best = None
    for m in (pen_blue, dark):
        prof = np.convolve(m.sum(1).astype(float), np.ones(9) / 9, "same") * prior
        if prof.max() >= 6:
            best = int(prof.argmax())
            break
    if best is None:
        return None
    a, b = int(max(0, best - 0.55 * box_h)), int(min(H, best + 0.5 * box_h))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # barcode in the band -> the wrong line was found: give up (reported as "Can't Read") rather than read the bars
    gb = g[a:b]
    if gb.size:
        bw = cv2.adaptiveThreshold(gb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, int(0.5 * (b - a))))))
        if (v.sum(0) > 0).mean() > 0.12:
            return None
    return g[a:b] if b - a >= 8 else None
