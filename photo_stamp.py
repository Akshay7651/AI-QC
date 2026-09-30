"""Burned-in GPS-camera stamp reader ("Latitude: .. Longitude: .. GPS Accuracy: .. Date: dd-mm-yyyy hh:mm AM").

The stamp is near-white text on a darker band at the bottom-left of a 720x1280 photo. We threshold the
near-white pixels (kills the photo background), upscale, stack the crops of all photos of a row into ONE
Tesseract call (OMP_THREAD_LIMIT=1, digits/letters whitelist, psm 6) and parse with tolerant regexes.
Falls back to per-photo retries with other thresholds and to 90/180/270-degree rotations (sideways stamp).
"""
import os, re, subprocess, tempfile
from datetime import datetime

import cv2
import numpy as np

import config as C

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
WL = "0123456789.:-LatitudeLongGPSAcrcyDmPM()/ "
_CFG = f"--psm 6 -c tessedit_char_whitelist={WL} -c load_system_dawg=0 -c load_freq_dawg=0"


def _mask(g, thr=235, frac=(0.80, 1.0, 0.0, 0.66), scale=2.0):
    h, w = g.shape
    c = g[int(frac[0] * h):int(frac[1] * h), int(frac[2] * w):int(frac[3] * w)]
    m = (c >= thr).astype(np.uint8) * 255
    m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return 255 - cv2.GaussianBlur(m, (3, 3), 0)


def _ocr(img):
    """img: 2-D uint8 array -> text (single Tesseract call)."""
    import pytesseract
    return pytesseract.image_to_string(img, lang="eng", config=_CFG, timeout=20)


def _num_after(line):
    s = line.split(":", 1)[1] if ":" in line else re.sub(r"^[A-Za-z()]+", "", line)
    s = re.sub(r"[^0-9.]", "", s)
    return s


def _coord(s, lo, hi):
    """'29.4106963' or '281246425' (lost dot) -> float within [lo,hi] or None. Returns (value, ndec)."""
    if not s:
        return None, 0
    s = s.strip(".")
    cands = []
    if "." in s:
        a, _, b = s.partition(".")
        b = b.replace(".", "")
        cands.append((a, b))
    else:
        cands.append((s[:2], s[2:]))
    for a, b in cands:
        if 2 <= len(a) <= 3 and len(b) >= 3:
            try:
                v = float(f"{a}.{b}")
            except ValueError:
                continue
            if lo <= v <= hi:
                return v, len(b)
    return None, 0


def parse_block(txt):
    """Parse one stamp's OCR text (spaces may be missing)."""
    out = {"lat": None, "lng": None, "date": None, "time": None, "acc": None, "lat_dec": 0, "lng_dec": 0}
    t = txt.replace("\n", "\n")
    for line in t.splitlines():
        l = line.strip()
        if not l:
            continue
        low = l.lower()
        if out["lat"] is None and ("lat" in low[:5] or "itude" in low and "long" not in low):
            out["lat"], out["lat_dec"] = _coord(_num_after(l), C.INDIA_LAT_MIN, C.INDIA_LAT_MAX)
        elif out["lng"] is None and ("long" in low[:6] or "ngitude" in low):
            out["lng"], out["lng_dec"] = _coord(_num_after(l), C.INDIA_LNG_MIN, C.INDIA_LNG_MAX)
        elif "acc" in low and out["acc"] is None:
            m = re.search(r"(\d+(?:\.\d+)?)\(?", l.split(":", 1)[-1])
            if m:
                try:
                    out["acc"] = float(m.group(1))
                except ValueError:
                    pass
        elif "date" in low or re.search(r"\d{2}-\d{2}-20\d{2}", l):
            m = re.search(r"(\d{2})-(\d{2})-(20\d{2})\s*(\d{1,2})?:?(\d{2})?\s*(AM|PM)?", l.replace("O", "0"))
            if m:
                try:
                    d = datetime(int(m[3]), int(m[2]), int(m[1]))
                    if 2024 <= d.year <= 2027:
                        out["date"] = d.strftime("%d%m%Y")
                        if m[4] and m[5] and m[6]:
                            hh = int(m[4]) % 12 + (12 if m[6] == "PM" else 0)
                            if int(m[5]) < 60 and 1 <= int(m[4]) <= 12:
                                out["time"] = f"{hh:02d}:{int(m[5]):02d}"
                except ValueError:
                    pass
    return out


def _ok(p):
    return p["lat"] is not None and p["lng"] is not None


def _stack(masks):
    w = max(m.shape[1] for m in masks)
    rows = []
    for m in masks:
        pad = np.full((m.shape[0], w - m.shape[1]), 255, np.uint8)
        rows.append(np.hstack([m, pad]))
        rows.append(np.full((40, w), 255, np.uint8))
    return np.vstack(rows)


def _split_blocks(txt):
    blocks, cur = [], []
    for line in txt.splitlines():
        if re.match(r"\s*(L[a-z]*t|Latit)", line) and cur and any(re.match(r"\s*L", c) for c in cur):
            if any(("lat" in c.lower()[:5]) for c in cur):
                blocks.append("\n".join(cur)); cur = []
        cur.append(line)
    blocks.append("\n".join(cur))
    return blocks


def read_stamps(grays):
    """grays: list of 2-D uint8 arrays (full-res photos). -> list of dicts (+ 'rot' = degrees needed, 'stamp_ok')."""
    n = len(grays)
    res = [None] * n
    try:
        masks = [_mask(g) for g in grays]
        txt = _ocr(_stack(masks))
        blocks = [b for b in _split_blocks(txt) if "lat" in b.lower()[:200]]
        if len(blocks) == n:
            for i, b in enumerate(blocks):
                res[i] = parse_block(b)
    except Exception:
        pass
    for i in range(n):
        if res[i] is not None and _ok(res[i]):
            res[i]["rot"] = 0
            continue
        best = res[i]
        # retries: other thresholds / crops, per photo
        for thr, sc, fr in ((215, 2.5, (0.80, 1.0, 0, 0.66)), (245, 2.0, (0.78, 1.0, 0, 0.75)), (200, 3.0, (0.75, 1.0, 0, 0.7))):
            try:
                p = parse_block(_ocr(_mask(grays[i], thr, fr, sc)))
            except Exception:
                continue
            if best is None or sum(v is not None for v in (p["lat"], p["lng"], p["date"])) > sum(v is not None for v in (best["lat"], best["lng"], best["date"])):
                best = p
            if _ok(p):
                break
        if best is not None and _ok(best):
            best["rot"] = 0
        else:
            # sideways / upside-down stamp
            for k, code in ((1, cv2.ROTATE_90_CLOCKWISE), (3, cv2.ROTATE_90_COUNTERCLOCKWISE), (2, cv2.ROTATE_180)):
                try:
                    p = parse_block(_ocr(_mask(cv2.rotate(grays[i], code))))
                except Exception:
                    continue
                if _ok(p):
                    p["rot"] = {1: 90, 3: 270, 2: 180}[k]
                    best = p
                    break
        if best is None:
            best = parse_block("")
        best.setdefault("rot", 0)
        res[i] = best
    for r in res:
        r["stamp_ok"] = _ok(r)
    return res
