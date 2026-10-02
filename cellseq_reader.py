"""Digit-by-digit reader for the handwritten AREA % / LOSS % table cells (Haryana Proforma-3).

    read_value(grey_or_rgb_cell) -> dict(value=float|None, text=str, conf=float)

Same network as the date / PO-ID readers (CNN + BiGRU + CTC, numpy inference) but on a narrower input; the classes are
blank, the digits 0-9 and '%'. It reads the characters in sequence ("9","0","%") instead of picking one of 23 whole-cell
classes, which is what the older cell model (digits.py) does. Trained by train_cellseq.py.
conf = the lowest per-character confidence. value is None when the text is not a plausible percentage (0..100).
"""
import os

import numpy as np

import date_reader as DR

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "cellseq_crnn.npz")
IH, IW = 48, 160
_M = None


def prep(g):
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


def parse(text):
    """'90%' / '090' / '00' / '100' -> float or None"""
    t = text.replace("-", "").rstrip("%")
    if not t or not t.isdigit() or "%" in t:
        return None
    v = int(t)
    return float(v) if 0 <= v <= 100 and len(t) <= 3 else None


def read_value(img):
    M = _load()
    if M is None:
        return {"value": None, "text": "", "conf": 0.0}
    g = DR.to_gray(img)
    P = DR.forward(prep(g), M)
    text, confs = DR.ctc_greedy(P)
    text = text.replace("-", "%")          # class 11 is '%' in this model
    return {"value": parse(text), "text": text, "conf": float(min(confs)) if confs else 0.0}
