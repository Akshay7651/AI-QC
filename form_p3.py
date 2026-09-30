"""Proforma-3 (PMFBY crop-loss assessment report) layout reader.

Uses the PRINTED Hindi text (reliable with Tesseract) as anchors, then crops the handwritten parts:
  - total row cells: affected area (%) and crop loss (%)
  - four signature boxes: farmer, loss-assessor company, primary worker, block agriculture officer
Returns crops + ink measurements; digit reading lives in digits.py (Tesseract digit mode / trained model).
"""
import io
import re

import numpy as np
from PIL import Image

import local_ocr

TARGET_W = 1600


def render(path, page=0):
    import pymupdf
    if str(path).lower().endswith(".pdf"):
        with pymupdf.open(str(path)) as d:
            pg = d[page]
            img = Image.open(io.BytesIO(pg.get_pixmap(dpi=int(72 * TARGET_W / pg.rect.width)).tobytes("png")))
    else:
        img = Image.open(path)
        img = img.resize((TARGET_W, int(img.height * TARGET_W / img.width)))
    return img.convert("L")


def anchors(img):
    W, H = img.size
    words = local_ocr._ocr(img)
    hdr = sorted([w for w in words if "प्रभावित" in w["t"] and w["x"] > 0.45 * W and 0.4 * H < w["y"] < 0.62 * H], key=lambda w: w["x"])
    tot = [w for w in words if "बीमित" in w["t"] and w["x"] < 0.3 * W and w["y"] > 0.6 * H]
    kul = [w for w in words if "कुल" in w["t"] and w["y"] > 0.6 * H and w["x"] < 0.3 * W]
    sig = sorted([w for w in words if "हस्ताक्षर" in w["t"] and w["y"] > 0.75 * H], key=lambda w: w["x"])
    return {"hdr": hdr, "total_row_y": (kul or tot or [None])[0]["y"] if (kul or tot) else None, "sig": sig, "W": W, "H": H}


def crops(img, a):
    """-> dict name -> PIL crop (or None when anchors are missing)."""
    W, H = a["W"], a["H"]
    out = {"area": None, "loss": None, "sig_farmer": None, "sig_company": None, "sig_worker": None, "sig_officer": None}
    if len(a["hdr"]) >= 2 and a["total_row_y"]:
        x1, x2 = a["hdr"][0]["x"], a["hdr"][1]["x"]
        y = a["total_row_y"]
        out["area"] = img.crop((x1 - 15, y - 15, x2 - 25, y + 70))
        out["loss"] = img.crop((x2 - 25, y - 15, min(W, x2 + 190), y + 70))
    sig = a["sig"]
    # keep one label per column: labels are spaced ~W/4 apart
    cols = []
    for w in sig:
        if not cols or w["x"] - cols[-1]["x"] > 0.12 * W:
            cols.append(w)
    names = ["sig_farmer", "sig_company", "sig_worker", "sig_officer"]
    for i, w in enumerate(cols[:4]):
        nx = cols[i + 1]["x"] - 20 if i + 1 < len(cols) else W
        out[names[i]] = img.crop((max(0, w["x"] - 30), max(0, w["y"] - 150), min(W, nx), w["y"] - 8))
    return out


def ink(crop, min_len_frac=0.25):
    """Fraction of handwriting-like dark pixels (printed straight lines removed)."""
    import cv2
    a = np.asarray(crop)
    if a.size == 0:
        return 0.0
    b = cv2.adaptiveThreshold(a, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    h, w = b.shape
    hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * min_len_frac)), 1)))
    vl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * min_len_frac)))))
    b = cv2.subtract(b, cv2.bitwise_or(hl, vl))
    return float((b > 0).mean())
