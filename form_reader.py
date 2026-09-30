"""Proforma-3 reader: read_form(path_or_pil, docket=None) -> dict  (see docs/agent_brief.md for the interface).

Geometry lives in form_p3.py; handwriting is read by digits.py (DIGITS agent)."""
import os
import re
import time

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import form_p3 as P  # noqa: E402

FORMNO_RE = re.compile(r"HR0126\d{6}")
FORMNO_LOOSE = re.compile(r"[H4K]R0126\d{6}")
DEBUG = []


# ----------------------------------------------------------------------------------------- FORM NO
def _bar_bbox(g, kx):
    """bounding box of the barcode in a gray window: largest wide blob of 'vertical bar' energy"""
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    e = cv2.boxFilter(gx, -1, (kx, 5)) - 2.0 * cv2.boxFilter(gy, -1, (kx, 5))
    s = float(e.max())
    if s < 12:
        return None
    m = (e > 0.25 * s).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for l in range(1, n):
        x0, y0, w, h = st[l, :4]
        if w < 4 * h or h < 6:
            continue
        if best is None or w * h > best[4]:
            best = (int(x0), int(y0), int(x0 + w), int(y0 + h), int(w * h))
    return best


def _tess(img, psm, binar):
    import pytesseract
    if binar == "otsu":
        _, o = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        o = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 14)
    o = cv2.copyMakeBorder(o, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    d = pytesseract.image_to_data(o, lang="eng", config=f"--psm {psm} -c tessedit_char_whitelist=HR0123456789",
                                  output_type=pytesseract.Output.DICT)
    words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip()]
    txt = " ".join(t for t, _ in words)
    cf = [c for _, c in words if c >= 0]
    return txt, (float(np.mean(cf)) / 100.0 if cf else 0.0)


def _find_formno(txt):
    t = txt.replace(" ", "")
    m = FORMNO_RE.search(t)
    if m:
        return m.group(0)
    m = FORMNO_LOOSE.search(t)
    if m:
        return "HR" + m.group(0)[2:]
    m = re.search(r"R(\d{9,10})", t)            # a dropped leading zero after the 'HR'
    if m and m.group(1).startswith(("126", "0126")):
        d = m.group(1)
        d = d if d.startswith("0") else "0" + d
        return "HR" + d if len(d) == 10 else None
    return None


def _ocr_formno(band, strip):
    """-> (form_no|None, conf, raw, note); several independent passes must agree for high confidence"""
    tries = [(band, 7, "otsu"), (strip, 6, "otsu"), (band, 7, "adapt"), (strip, 6, "adapt")]
    got = []
    raws = []
    for img, psm, bn in tries:
        txt, cf = _tess(img, psm, bn)
        raws.append(txt)
        f = _find_formno(txt)
        if f:
            got.append((f, cf))
            if len(got) >= 2 and got[0][0] == got[1][0]:
                break
        if len(got) == 2 and got[0][0] != got[1][0]:
            continue
    if not got:
        return None, 0.0, raws[0], "form no pattern not matched"
    from collections import Counter
    cnt = Counter(f for f, _ in got)
    f, n = cnt.most_common(1)[0]
    cf = max(c for ff, c in got if ff == f)
    if n >= 2:
        return f, min(1.0, 0.9 + 0.1 * cf), raws[0], ""
    if len(cnt) > 1:
        return f, 0.4, raws[0], "form no passes disagree"
    return f, 0.6 + 0.2 * cf, raws[0], "form no from a single pass"


def read_formno(lay):
    """-> (form_no|None, conf, raw_text, note)"""
    r, B = lay["rgb"], lay["boxes"]
    if "formno_win" not in B:
        return None, 0.0, "", "no form-no window"
    p = lay["tab"]["p"] * lay["scale"]
    x0, y0, x1, y1 = B["formno_win"]
    wx0, wx1 = int(max(0, x0 - 0.1 * (x1 - x0))), int(min(r.shape[1], x1 + 0.15 * (x1 - x0)))
    wy0, wy1 = int(max(0, y0 - 1.5 * p)), int(min(r.shape[0], y1 + 2.2 * p))
    if wy1 - wy0 < 10 or wx1 - wx0 < 10:
        return None, 0.0, "", "form-no window outside image"
    g = cv2.cvtColor(r[wy0:wy1, wx0:wx1], cv2.COLOR_RGB2GRAY)
    bb = _bar_bbox(g, max(9, int(1.3 * p)))
    if bb is None:
        return None, 0.0, "", "barcode not found"
    bx0, by0, bx1, by1, _ = bb
    bh, bw = by1 - by0, bx1 - bx0
    ya, yb = by1 + int(0.02 * bh), min(g.shape[0], by1 + int(1.25 * bh))
    xa, xb = max(0, bx0 - int(0.12 * bw)), min(g.shape[1], bx1 + int(0.2 * bw))
    if yb - ya < 8:
        return None, 0.0, "", "no room under barcode"
    st = g[ya:yb, xa:xb]
    sc = max(2.0, 70.0 / max(1, bh))
    st = cv2.resize(st, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    bh2 = bh * sc
    bn = cv2.adaptiveThreshold(st, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 14)
    n, lab, stt, cen = cv2.connectedComponentsWithStats(bn)
    comps = []
    for l in range(1, n):
        x, y, w, h, ar = stt[l]
        if 0.22 * bh2 < h < 0.8 * bh2 and w < 1.4 * h and ar > 0.12 * h * h * 0.3:
            comps.append((x, y, w, h))
    band = None
    if len(comps) >= 6:
        comps.sort(key=lambda c: c[1] + c[3] / 2)
        groups, cur = [], [comps[0]]
        for c in comps[1:]:
            if abs((c[1] + c[3] / 2) - np.mean([k[1] + k[3] / 2 for k in cur])) < 0.22 * bh2:
                cur.append(c)
            else:
                groups.append(cur)
                cur = [c]
        groups.append(cur)
        groups = [g_ for g_ in groups if len(g_) >= 8]
        if groups:
            gg = groups[0]                    # closest to the barcode
            x0_ = min(c[0] for c in gg)
            x1_ = max(c[0] + c[2] for c in gg)
            y0_ = min(c[1] for c in gg)
            y1_ = max(c[1] + c[3] for c in gg)
            pad = int(0.2 * (y1_ - y0_)) + 3
            band = st[max(0, y0_ - pad): y1_ + pad, max(0, x0_ - pad): x1_ + pad]
    if band is None:
        return None, 0.0, "", "form-no text line not found"
    DEBUG.append((g, (bx0, by0, bx1, by1), st, band))
    return _ocr_formno(band, st)
