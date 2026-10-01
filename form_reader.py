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
def _bar_bbox(g, kx, hmin=0, hmax=10 ** 9):
    """bounding box of the barcode in a gray window: best wide blob of 'vertical bar' energy (bars have strong
    horizontal gradients and weak vertical ones; handwriting / underlines do not)"""
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    e = cv2.boxFilter(gx, -1, (kx, 5)) - 2.0 * cv2.boxFilter(gy, -1, (kx, 5))
    s = float(e.max())
    if s < 10:
        return None
    m = (e > 0.3 * s).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    best = None
    for l in range(1, n):
        x0, y0, w, h = st[l, :4]
        if w < 3 * h or h < max(6, hmin) or h > hmax:
            continue
        sc = float(e[lab == l].mean()) * w
        if best is None or sc > best[4]:
            best = (int(x0), int(y0), int(x0 + w), int(y0 + h), sc)
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
    """-> (form_no|None, conf, raw, note). Up to 9 differently pre-processed Tesseract passes vote on the string.

    Measured on 135 hand-labelled forms (9 passes): >=3 passes agreeing -> 96.0% exact (73% of forms),
    >=4 -> 97.8% (69%), >=5 -> 98.8% (64%), >=6 -> 100% (56%); best-guess with any agreement -> 92% (83%).
    So form_no_conf >= 0.85 means '>=3 passes agree' (~96% exact); below that the value is only a hint.
    """
    from collections import Counter
    sharp = cv2.addWeighted(strip, 2.0, cv2.GaussianBlur(strip, (0, 0), 3), -1.0, 0)
    big = cv2.resize(strip, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    tries = []
    if band is not None:
        tries += [(band, 7, "otsu"), (band, 7, "adapt")]
    tries += [(strip, 11, "otsu"), (strip, 6, "otsu"), (sharp, 11, "otsu"), (strip, 11, "adapt"),
              (big, 11, "otsu"), (big, 6, "otsu"), (sharp, 6, "otsu")]
    got, raws = [], []
    for img, psm, bn in tries:
        txt, cf = _tess(img, psm, bn)
        raws.append(txt)
        f = _find_formno(txt)
        if f:
            got.append((f, cf))
            if Counter(x for x, _ in got).most_common(1)[0][1] >= 4:   # 4 agreeing passes: stop early (97.8% exact)
                break
    if not got:
        return None, 0.0, raws[0] if raws else "", "form no pattern not matched"
    cnt = Counter(f for f, _ in got)
    f, n = cnt.most_common(1)[0]
    conf = {1: 0.6, 2: 0.75, 3: 0.93, 4: 0.97, 5: 0.98}.get(n, 0.99)
    note = "" if n >= 3 else ("form no: only %d agreeing pass(es) - verify manually" % n)
    if len(cnt) > 1 and n < 3:
        note = "form no passes disagree - verify manually"
    return f, conf, raws[0], note


def _formno_scan(r):
    """Fallback when the barcode/under-barcode window was not found (tilted, shadowed or off-centre photos): scan the whole top-right
    of the page.  Same voting rule as _ocr_formno (>= 3 agreeing passes are needed to be asserted)."""
    from collections import Counter
    H, W = r.shape[:2]
    got = []
    for fy in (0.17, 0.24):
        g = cv2.cvtColor(r[: int(fy * H), int(0.42 * W):], cv2.COLOR_RGB2GRAY)
        sc = 1500.0 / max(1, g.shape[1])
        g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
        sharp = cv2.addWeighted(g, 2.0, cv2.GaussianBlur(g, (0, 0), 3), -1.0, 0)
        for img, psm, bn in ((g, 6, "otsu"), (g, 6, "adapt"), (sharp, 6, "otsu"), (g, 11, "otsu")):
            try:
                txt, _ = _tess(img, psm, bn)
            except Exception:     # noqa: BLE001
                continue
            f = _find_formno(txt)
            if f:
                got.append(f)
                if Counter(got).most_common(1)[0][1] >= 3:
                    break
        if got and Counter(got).most_common(1)[0][1] >= 3:
            break
    if not got:
        return None, 0.0, "", "form no pattern not matched"
    cnt = Counter(got)
    f, n = cnt.most_common(1)[0]
    conf = {1: 0.6, 2: 0.75, 3: 0.93, 4: 0.97, 5: 0.98}.get(n, 0.99)
    note = "" if n >= 3 else ("form no: only %d agreeing pass(es) - verify manually" % n)
    return f, conf, "", note


def read_formno(lay):
    """-> (form_no|None, conf, raw_text, note)"""
    r, sc = lay["rgb"], lay["scale"]
    tab = lay["tab"]
    gb = lay.get("gb")
    p = tab["p"] * sc
    w = (tab["R"] - tab["L"]) * sc
    T = (gb["T"] if gb else tab["ytop"] - (P.GRID_GAP + P.GRID_H) * tab["p"]) * sc
    x0 = int(max(0, tab["L"] * sc + 0.3 * w))
    x1 = int(min(r.shape[1], tab["R"] * sc + 0.25 * w))
    y0 = 0
    y1 = int(0.45 * r.shape[0])
    if y1 - y0 < 10 or x1 - x0 < 10:
        return None, 0.0, "", "form-no window outside image"
    g = cv2.cvtColor(r[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    bb = _bar_bbox(g, max(9, int(1.3 * p)), hmin=int(0.3 * p), hmax=int(3.0 * p))
    if bb is None:
        return None, 0.0, "", "barcode not found"
    bx0, by0, bx1, by1, _ = bb
    bh, bw = by1 - by0, bx1 - bx0
    ya, yb = by1 - int(0.05 * bh), min(g.shape[0], by1 + int(1.35 * bh))
    xa, xb = max(0, bx0 - int(0.25 * bw)), min(g.shape[1], bx1 + int(0.3 * bw))
    if yb - ya < 8:
        return None, 0.0, "", "no room under barcode (page cropped?)"
    st = g[ya:yb, xa:xb]
    scl = max(2.0, 70.0 / max(1, bh))
    st = cv2.resize(st, None, fx=scl, fy=scl, interpolation=cv2.INTER_CUBIC)
    bh2 = bh * scl
    bn = cv2.adaptiveThreshold(st, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 14)
    n, lab, stt, cen = cv2.connectedComponentsWithStats(bn)
    comps = []
    for l in range(1, n):
        x, y, w_, h_, ar = stt[l]
        if 0.22 * bh2 < h_ < 0.8 * bh2 and w_ < 1.4 * h_ and y > 0.08 * bh2:
            comps.append((x, y, w_, h_))
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
            gg = groups[0]
            x0_, x1_ = min(c[0] for c in gg), max(c[0] + c[2] for c in gg)
            y0_, y1_ = min(c[1] for c in gg), max(c[1] + c[3] for c in gg)
            pad = int(0.2 * (y1_ - y0_)) + 3
            band = st[max(0, y0_ - pad): y1_ + pad, max(0, x0_ - pad): x1_ + pad]
    DEBUG.append((g, (bx0, by0, bx1, by1), st, band))
    return _ocr_formno(band, st)


# ----------------------------------------------------------------------------------------- signatures / stamp
def _chroma_rel(rgb):
    """(B - R) relative to the crop's own median -> robust to bluish/yellowish lighting"""
    c = rgb.astype(np.int16)
    d = c[..., 2] - c[..., 0]
    return d - int(np.median(d))


def sig_features(rgb_box, p, split=1.1):
    """features of a signature box crop (RGB, hi-res; its top edge is 1.0 pitch below the table). p = row pitch in px.
    area_low = ink area (in p^2) below `split` pitches from the crop top (where signatures sit); area_top = above
    (where handwritten committee remarks sit)."""
    z = {"area": 0.0, "area_low": 0.0, "area_top": 0.0, "ncomp": 0, "ext_w": 0.0, "ext_h": 0.0, "blue": 0.0, "cy": 0.0}
    if rgb_box.size == 0 or min(rgb_box.shape[:2]) < 5:
        return z
    m = P.ink_mask(rgb_box, min_len_frac=0.35, contrast=20)
    chroma = rgb_box.astype(np.int16)
    red_print = ((chroma[..., 0] - chroma[..., 2]) > 18) & (chroma.sum(2) < 520)     # printed brown/red label text
    m[red_print] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(m, np.ones((3, 3), np.uint8)))
    keep = [l for l in range(1, n) if st[l, 4] >= 0.02 * p * p]
    if not keep:
        return z
    sel = np.isin(lab, keep) & (m > 0)
    ys_, xs_ = np.nonzero(sel)
    cut = int(split * p)
    z["area"] = float(sel.sum()) / (p * p)
    z["area_top"] = float(sel[:cut].sum()) / (p * p)
    z["area_low"] = float(sel[cut:].sum()) / (p * p)
    z["ncomp"] = len(keep)
    z["ext_w"] = float(xs_.max() - xs_.min()) / rgb_box.shape[1]
    z["ext_h"] = float(ys_.max() - ys_.min()) / p
    z["cy"] = float(ys_.mean()) / p
    z["blue"] = float((_chroma_rel(rgb_box)[sel] > 12).mean())
    return z


def stamp_features(rgb_zone, p):
    """printed rubber stamp (blue/violet regular text) in the zone around the officer label"""
    if rgb_zone.size == 0 or min(rgb_zone.shape[:2]) < 5:
        return {"frac": 0.0, "span": 0.0}
    m = P.ink_mask(rgb_zone, min_len_frac=0.5, contrast=12)
    d = _chroma_rel(rgb_zone)
    bm = (m > 0) & (d > 14)
    if bm.sum() < 0.05 * p * p:
        return {"frac": float(bm.mean()), "span": 0.0}
    cols = np.where(bm.sum(0) > 0)[0]
    return {"frac": float(bm.mean()), "span": float((cols.max() - cols.min()) / rgb_zone.shape[1])}


SIG_LOW_MIN = {"farmer": 0.40, "company": 0.20, "worker": 0.30, "officer": 0.90}   # ink area (p^2 units) below the remark zone
STAMP_FRAC_MIN = 0.012
STAMP_SPAN_MIN = 0.45


def read_signatures(lay):
    r, B, sc = lay["rgb"], lay["boxes"], lay["scale"]
    p = lay["tab"]["p"] * sc
    yt = lay["tab"]["ytot"] * sc
    out, feats = {}, {}
    for nm in ("farmer", "company", "worker", "officer"):
        x0, _, x1, y1 = B["sig_" + nm]
        f = sig_features(r[int(max(0, yt + 1.0 * p)): int(y1), int(max(0, x0)): int(x1)], p)
        feats[nm] = f
        out[nm] = bool(f["area_low"] >= SIG_LOW_MIN[nm] and (f["ext_w"] >= 0.12 or f["ext_h"] >= 0.45))
    x0, _, x1, _ = B["sig_officer"]
    zone = r[int(yt + 3.3 * p): int(min(r.shape[0], yt + 5.4 * p)), int(x0): int(x1)]
    sf = stamp_features(zone, p)
    feats["stamp"] = sf
    stamp = bool(sf["frac"] >= STAMP_FRAC_MIN and sf["span"] >= STAMP_SPAN_MIN)
    return out, stamp, feats


# ----------------------------------------------------------------------------------------- learned signature classifier
_SIGCLF = None


def _sig_clf():
    """models/sig_clf.joblib (tools/train_sig_classifier.py) or None -> the old ink rule stays in charge."""
    global _SIGCLF
    if _SIGCLF is None:
        try:
            import joblib
            _SIGCLF = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sig_clf.joblib"))
        except Exception:      # noqa: BLE001
            _SIGCLF = False
    return _SIGCLF or None


def _sig_pred(blk, feats):
    """-> bool | None (None = no classifier / no features)"""
    clf = _sig_clf()
    if not clf or not feats or not feats.get(blk) or blk not in clf["models"]:
        return None
    m, f = clf["models"][blk], feats[blk]
    x = [f[k] for k in clf["keys"]]
    if isinstance(m, dict):                       # 'common answer unless ink far below normal'
        low = x[clf["keys"].index(m["key"])] < m["thr"]
        return bool(1 - m["common"]) if low else bool(m["common"])
    return bool(m.predict([x])[0])


# ----------------------------------------------------------------------------------------- digits.py bridge
_DIG = None


def _digits():
    """digits.py module if importable AND its trained model exists, else None (handwriting then stays unread)"""
    global _DIG
    if _DIG is None:
        try:
            import digits as D
            _DIG = D
        except Exception:      # noqa: BLE001
            return None
    mp = getattr(_DIG, "MODEL_PATH", None)
    if mp is not None and not os.path.exists(mp):
        return None
    return _DIG


def _pil(a):
    from PIL import Image
    return Image.fromarray(a if a.ndim == 2 else a)


def _gray_hi(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb


def read_cell(rgb_crop):
    """-> dict(value, text, conf, ink, blank)"""
    ink = P.ink_frac(rgb_crop, inset=0.08)
    blank = ink < CELL_BLANK_INK
    out = {"value": None, "text": "", "conf": 0.0, "ink": ink, "blank": blank}
    D = _digits()
    if blank or D is None:
        return out
    try:
        res = D.read_number(_pil(_gray_hi(rgb_crop)), max_digits=3, allow_decimal=True)
        out.update(value=res.get("value"), text=res.get("text", ""), conf=float(res.get("conf", 0.0)))
    except Exception as e:     # noqa: BLE001
        out["err"] = str(e)
    return out


FORMNO_GATE = 0.9   # >=3 agreeing OCR passes (see _ocr_formno)
CELL_BLANK_INK = 0.006
# Handwritten area/loss values are only asserted when the cell model's confidence is >= CELL_GATE.
# Retrained model (human disputed-case labels), 114 never-trained forms: gate 0.7 -> 96.9% precision, 71% of cells answered (non-zero answered: 95% right).
CELL_GATE = 0.85
# Dates: sequence reader (date_reader.py: CNN+BiGRU+CTC, numpy). Switched on only when its measured precision-at-gate is >=95%
# per field on held-out forms (see docs / train_dates.py --eval); False = dates stay None ("not readable").
# Measured on held-out forms (precision when answered / coverage): loss date 95-97% / 45-62% -> ON with a strict gate;
# intimation 73-90%, inspection 67-79%, sowing 43-74% -> OFF (weak labels: the app intimation date equals the form's only 19% of the time;
# month-name dates; rare sowing months). Turn a field on only after it reaches >=95% on held-out human-labelled dates.
DATES_ENABLED = {"loss_date": True, "sow_date": False, "intimation_date": False, "inspection_date": False}
DATE_GATE = {"loss_date": 0.95}
DATE_FIELD = {"sow_date": "sow", "loss_date": "loss", "intimation_date": "intim", "inspection_date": "insp"}


def read_date_box(rgb_crop, field, gate=0.0):
    """handwritten date box -> (DDMMYYYY|None, conf, note); blank box -> (None, 1.0, 'blank'); below the gate -> (None, conf, note)"""
    import date_reader as DR
    r = DR.read_date(rgb_crop, field)
    if r.get("blank"):
        return None, 1.0, "blank"
    if r["date"] is None:
        return None, 0.0, ("ambiguous date '%s'" % r["text"]) if r.get("ambiguous") else ("date '%s' not parsable" % r["text"])
    if r["conf"] < max(DR.CONF_GATE, gate):
        return None, float(r["conf"]), "date '%s' low confidence" % r["text"]
    return r["date"], float(r["conf"]), ""


def _digit_probs(rgb_crop):
    """per-component class probabilities (digits.py internals) -> (chars list [(ch, prob, probs13)], comps) or None"""
    D = _digits()
    if D is None or not hasattr(D, "_analyse") or not hasattr(D, "classify_components"):
        return None
    try:
        comps, H, W = D._analyse(_pil(_gray_hi(rgb_crop)))
        comps = [c for c in comps if not c.get("dot")]
        if not comps:
            return [], comps
        probs = D.classify_components(comps)
        return probs, comps
    except Exception:      # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------- dates
def parse_date(rgb_crop):
    """handwritten date cell -> (DDMMYYYY|None, conf, note). Uses digits.py component classes: digits + separator (slash/dash) class."""
    ink = P.ink_frac(rgb_crop, inset=0.08)
    if ink < CELL_BLANK_INK:
        return None, 1.0, "blank"
    pr = _digit_probs(rgb_crop)
    D = _digits()
    groups = None
    conf = 0.0
    if pr is not None and len(pr[1]):
        probs, comps = pr
        chars = []
        for c, pv in zip(comps, probs):
            k = int(np.argmax(pv))
            chars.append((k, float(pv[k]), c))
        cur, groups, confs = "", [], []
        H = max(c["y1"] - c["y0"] for _, _, c in chars)
        for k, pk, c in chars:
            ch = c["y1"] - c["y0"]
            cw = c["x1"] - c["x0"]
            sep = (k >= 10) or (k == 1 and ch > 1.35 * np.median([cc["y1"] - cc["y0"] for _, _, cc in chars]) and cw < 0.5 * ch)
            if sep:
                if cur:
                    groups.append(cur)
                cur = ""
            else:
                cur += str(k)
                confs.append(pk)
        if cur:
            groups.append(cur)
        conf = float(np.mean(confs)) if confs else 0.0
    elif D is not None:
        res = D.read_digit_string(_pil(_gray_hi(rgb_crop)))
        t = res.get("text", "")
        conf = float(res.get("conf", 0.0))
        groups = [t] if t else None
    if not groups:
        return None, 0.0, "no digits"
    return _assemble_date(groups, conf)


def _assemble_date(groups, conf):
    import datetime
    d = m = y = None
    note = ""
    if len(groups) >= 3:
        d, m, y = groups[0], groups[1], groups[2]
        if len(groups) > 3:
            note = "extra date parts ignored"
    elif len(groups) == 1:
        t = groups[0]
        if len(t) == 8:
            d, m, y = t[:2], t[2:4], t[4:]
        elif len(t) == 6:
            d, m, y = t[:2], t[2:4], t[4:]
        else:
            return None, 0.0, f"date digits '{t}' unparsable"
    else:
        return None, 0.0, f"date parts {groups} unparsable"
    if len(y) == 2:
        y = "20" + y
    if len(y) == 1:
        y = "202" + y
    if len(y) == 3:                      # dropped digit, e.g. 026 / 226
        y = "2026" if y.endswith("26") else y
    if len(y) > 4:
        y = y[-4:]
    try:
        di, mi, yi = int(d), int(m), int(y)
        if yi in (2020, 2028, 2029, 2005):      # 2026 often misread: flag but keep
            note += " year unusual"
        datetime.date(yi, mi, di)
    except Exception:      # noqa: BLE001
        return None, 0.0, f"invalid date {d}/{m}/{y}"
    return f"{di:02d}{mi:02d}{yi:04d}", conf, note.strip()


# ----------------------------------------------------------------------------------------- PO ID
def read_po_id(rgb_crop, docket=None):
    """-> (po_id|None, conf, matches|None, info dict)"""
    info = {"n_comp": 0, "ink": P.ink_frac(rgb_crop, inset=0.05)}
    if info["ink"] < 0.004:
        return None, 1.0, (False if docket else None), dict(info, blank=True)
    pr = _digit_probs(rgb_crop)
    D = _digits()
    text, conf, probs = "", 0.0, None
    if pr is not None and len(pr[1]):
        probs, comps = pr
        # drop the printed 'PO ID:' remains / underline: keep components whose height is in digit range
        hs = np.array([c["y1"] - c["y0"] for c in comps], float)
        medh = np.median(hs)
        keep = [i for i, c in enumerate(comps) if hs[i] >= 0.5 * medh]
        probs = probs[keep]
        comps = [comps[i] for i in keep]
        info["n_comp"] = len(comps)
        dig = probs[:, :10]
        text = "".join(str(int(r.argmax())) for r in dig)
        conf = float(np.mean(dig.max(1) / np.maximum(dig.sum(1), 1e-9)))
    elif D is not None:
        res = D.read_digit_string(_pil(_gray_hi(rgb_crop)), expected_len=18)
        text, conf = res.get("text", ""), float(res.get("conf", 0.0))
        info["n_comp"] = res.get("n_components", 0)
    matches = None
    if docket:
        if text == docket:
            matches = True
        elif probs is not None and len(probs) == len(docket):
            # verification: is the docket digit the best / a plausible class at every position?
            pd = np.array([probs[i, int(ch)] / max(probs[i, :10].sum(), 1e-9) for i, ch in enumerate(docket)])
            if (pd >= 0.25).all():
                matches = True
            elif (pd < 0.05).any():
                matches = False
        elif len(text) >= 10 and len(text) != len(docket) and info["n_comp"] not in (len(docket) - 1, len(docket) + 1):
            matches = False if len(text) < 14 else None
        elif text and len(text) == len(docket):
            diff = sum(a != b for a, b in zip(text, docket))
            matches = False if diff >= 5 else None
    return (text or None), conf, matches, info


# ----------------------------------------------------------------------------------------- quality / identification
def _quality(lay_or_rgb):
    rgb = lay_or_rgb
    g = _gray_hi(rgb)
    gs = P._resize_w(g, 800)
    lap = cv2.Laplacian(gs, cv2.CV_32F).var()
    mean = float(gs.mean())
    return {"blur": float(lap), "mean": mean, "contrast": float(np.percentile(gs, 95) - np.percentile(gs, 5))}


def _hin_header_is_form(rgb):
    """Tesseract hin on a small strip (only used when the ruled-line layout is ambiguous)."""
    try:
        import pytesseract
        g = P._resize_w(_gray_hi(rgb), 900)
        strip = g[: int(0.3 * g.shape[0])]
        txt = pytesseract.image_to_string(strip, lang="hin+eng", config="--psm 6")
        hit = ("प्रारूप" in txt) or ("रिपोर्ट" in txt and "हानि" in txt) or ("PO ID" in txt and "मौसम" in txt)
        return bool(hit), txt[:80]
    except Exception:      # noqa: BLE001
        return False, ""


def _empty(notes):
    return {"is_proforma3": False, "quality": "unknown", "form_no": None, "form_no_conf": 0.0, "po_id": None,
            "po_id_matches": None, "form_area": None, "form_loss": None, "row_area": None, "row_loss": None,
            "total_row_blank": False, "sow_date": None, "loss_date": None, "intimation_date": None,
            "inspection_date": None, "farmer_signed": False, "company_signed": False, "worker_signed": False,
            "officer_signed": False, "officer_stamp_only": False, "overwrite_suspected": False, "confidence": 0.0,
            "field_conf": {}, "notes": list(notes)}


def read_form(path_or_pil, docket=None, debug=False):
    """Read one photographed Proforma-3. See module docstring / docs/agent_brief.md for the returned keys."""
    t0 = time.time()
    out = _empty([])
    notes = out["notes"]
    try:
        rgb = P.load_image(path_or_pil)
    except Exception as e:     # noqa: BLE001
        notes.append(f"image unreadable: {e}")
        out["quality"] = "unreadable"
        return out
    q = _quality(rgb)
    if q["mean"] < 60:
        out["quality"] = "dark"
    elif q["blur"] < 25:
        out["quality"] = "blurry"
    else:
        out["quality"] = "good"
    lay = P.analyse_layout(rgb)
    if lay.get("ok") and lay.get("rgb") is not None and not os.environ.get("AIQC_NO_RAJ"):
        try:
            import form_raj
            if form_raj.is_rajasthan(lay["rgb"]):
                out["quality_metrics"] = {k: round(v, 1) for k, v in q.items()}
                if form_raj.read_form_raj(lay, out, CELL_GATE, notes):
                    out["confidence"] = round(float(np.mean([c for c in out["field_conf"].values()] or [0.3])) * (1.0 if out["quality"] == "good" else 0.7), 3)
                    out["elapsed"] = round(time.time() - t0, 3)
                    return out
        except Exception as e:     # noqa: BLE001
            notes.append(f"Rajasthan reader failed: {type(e).__name__}")
    out["quality_metrics"] = {k: round(v, 1) for k, v in q.items()}
    out["rotation_deg"] = lay["info"].get("rot", 0) if "info" in lay else 0
    out["skew_deg"] = round(lay["info"].get("skew", 0.0), 2) if "info" in lay else 0.0
    if not lay["ok"]:
        ok, sample = _hin_header_is_form(lay.get("rgb", rgb)) if lay.get("info", {}).get("lines") else (False, "")
        out["is_proforma3"] = bool(ok)
        notes.extend(lay["notes"])
        if ok:
            notes.append("looks like a Proforma-3 but the table layout could not be located (cropped / heavily distorted)")
            out["quality"] = "cropped" if out["quality"] == "good" else out["quality"]
        else:
            notes.append("not recognised as a Proforma-3 form (no ruled table / printed header)")
        out["confidence"] = 0.0
        out["elapsed"] = round(time.time() - t0, 3)
        return out

    tab, B, r, sc = lay["tab"], lay["boxes"], lay["rgb"], lay["scale"]
    fc = out["field_conf"]
    out["is_proforma3"] = True
    if tab["n_cols"] < 5 or tab["n_rowlines"] < 8:
        notes.append("table only weakly matched to the template")
    grid_ok = bool(lay.get("grid") and lay["grid"].get("detected"))
    W = r.shape[1]
    if tab["L"] < 0.02 * P.GW or tab["R"] > 0.985 * P.GW:
        out["quality"] = "cropped" if out["quality"] == "good" else out["quality"]
        notes.append("table touches the image border (page may be cropped)")
    # ---- FORM NO
    try:
        fn, fcf, raw, note = read_formno(lay)
    except Exception as e:     # noqa: BLE001
        fn, fcf, raw, note = None, 0.0, "", f"form no reading failed: {type(e).__name__}"
    if not fn or float(fcf) < FORMNO_GATE:       # second chance: scan the top of the page
        try:
            fn2, fcf2, raw2, note2 = _formno_scan(r)
            if fn2 and float(fcf2) > float(fcf):
                fn, fcf, raw, note = fn2, fcf2, raw2, note2
        except Exception:     # noqa: BLE001
            pass
    out["form_no"], out["form_no_conf"] = fn, round(float(fcf), 3)
    out["form_no_guess"] = None
    if fn and float(fcf) < FORMNO_GATE:       # measured: only '>=3 agreeing passes' is ~96-97% exact -> weaker reads are hints, not facts
        out["form_no_guess"], out["form_no"] = fn, None
    fc["form_no"] = out["form_no_conf"]
    if note:
        notes.append(note)
    # ---- PO ID
    if "po_id" in B and P.crop(r, B["po_id"]).size:
        po, pcf, pm, pinfo = read_po_id(P.crop(r, B["po_id"]), docket)
        # PO-ID handwriting reading does not work yet (0 of 337 forms read fully): never assert a match or a mismatch.
        out["po_id"], out["po_id_matches"] = None, None
        fc["po_id"] = round(float(pcf), 3)
        if pinfo.get("blank"):
            notes.append("PO ID field appears blank")
        # (PO-ID stroke-count overwrite rule disabled: it fired on ~55% of forms because PO-ID segmentation is unreliable)
    # ---- table cells
    cells = {}
    for k in ("area_r1", "loss_r1", "area_r2", "loss_r2", "area_tot", "loss_tot"):
        cells[k] = read_cell(P.crop(r, B[k], 0 if k.endswith("_tot") else 2))
    out["_cells"] = {k: {kk: vv for kk, vv in v.items() if kk != "err"} for k, v in cells.items()}
    tot_blank = cells["area_tot"]["blank"] and cells["loss_tot"]["blank"]
    out["total_row_blank"] = bool(tot_blank)
    if not tot_blank:
        out["form_area"] = cells["area_tot"]["value"]
        out["form_loss"] = cells["loss_tot"]["value"]
        fc["form_area"], fc["form_loss"] = round(cells["area_tot"]["conf"], 3), round(cells["loss_tot"]["conf"], 3)
    for rr in ("r1", "r2"):
        if not (cells["area_" + rr]["blank"] and cells["loss_" + rr]["blank"]):
            out["row_area"], out["row_loss"] = cells["area_" + rr]["value"], cells["loss_" + rr]["value"]
            fc["row_area"], fc["row_loss"] = round(cells["area_" + rr]["conf"], 3), round(cells["loss_" + rr]["conf"], 3)
            break
    # ---- confidence gate on handwritten values: below the gate the value is withheld ('not readable - verify manually')
    low = []
    for key, which in (("form_area", "area_tot"), ("form_loss", "loss_tot")):
        if out[key] is not None and cells[which]["conf"] < CELL_GATE:
            out[key] = None
            low.append(which)
    for key, which in (("row_area", "area_"), ("row_loss", "loss_")):
        if out[key] is not None:
            for rr in ("r1", "r2"):
                c = cells[which + rr]
                if not c["blank"] and c["value"] == out[key] and c["conf"] < CELL_GATE:
                    out[key] = None
                    low.append(which + rr)
                    break
    if low:
        notes.append("handwritten area/loss not confidently readable (" + ", ".join(low) + ") - verify manually")
    out["cells_low_conf"] = bool(low)
    # ---- dates
    for key, box in (("sow_date", "sow_date"), ("loss_date", "loss_date"), ("intimation_date", "intim_date"), ("inspection_date", "insp_date")):
        if box in B and DATES_ENABLED.get(key):
            try:
                d, dc, dn = read_date_box(P.crop(r, B[box], 2), DATE_FIELD[key], DATE_GATE.get(key, 0.0))
            except Exception as e:     # noqa: BLE001
                d, dc, dn = None, 0.0, "date reading failed: %s" % type(e).__name__
            out[key] = d
            fc[key] = round(float(dc), 3)
            if dn and dn != "blank":
                notes.append(f"{key}: {dn}")
    # ---- signatures
    try:
        sg, stamp, feats = read_signatures(lay)
    except Exception as e:     # noqa: BLE001
        sg, stamp, feats = {"farmer": False, "company": False, "worker": False, "officer": False}, False, {}
        notes.append(f"signature analysis failed: {type(e).__name__}")
    out["farmer_signed"], out["company_signed"], out["worker_signed"], out["officer_signed"] = (sg["farmer"], sg["company"], sg["worker"], sg["officer"])
    for blk in ("farmer", "company", "worker"):          # learned classifier overrides the ink rule when available
        pv = _sig_pred(blk, feats)
        if pv is not None:
            out[blk + "_signed"] = pv
    # The block officer is almost never signed (1 positive in ~246 labelled forms), so a reliable reading is impossible:
    # report 'not assessed' instead of a misleading Yes/No.
    out["officer_signed"] = None
    out["officer_stamp_only"] = bool(stamp and not sg["officer"])
    out["_sig_feats"] = feats
    # ---- overwrite on cells
    for k, v in cells.items():
        if v["ink"] > 0.30:
            out["overwrite_suspected"] = True
            notes.append(f"{k}: very heavy ink (possible over-writing / scribble)")
            break
    # ---- is it really a Proforma-3? (ruled table alone is not enough: need the form no, the upper grid or a Hindi header)
    strong = tab["n_cols"] >= 6 and tab["n_rowlines"] >= 11
    if not fn and not grid_ok and not strong:
        ok_h, _ = _hin_header_is_form(r)
        if not ok_h:
            out["is_proforma3"] = False
            notes.append("ruled table found but no form number / field grid / printed header: probably not a Proforma-3")
    # ---- confidence
    parts = [out["form_no_conf"]] if fn else []        # an unreadable form number is reported separately, it must not sink the handwriting confidence
    parts += [c for k, c in fc.items() if k in ("form_area", "form_loss", "row_area", "row_loss")]
    conf = float(np.mean(parts)) if parts else 0.3
    if out["quality"] != "good":
        conf *= 0.7
    if not lay.get("grid") or not lay["grid"].get("detected"):
        conf *= 0.9
    out["confidence"] = round(conf, 3)
    out["elapsed"] = round(time.time() - t0, 3)
    if debug:
        out["_lay"] = lay
    return out
