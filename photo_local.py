"""Free, offline photo QC (no API): EXIF + burned-in stamp OCR + colour/vegetation + person detection.

Cheap heuristics - every result is low/medium confidence, so field_photo / loss estimates are hints
for the human reviewer, not verdicts. GPS/date checks (EXIF or stamp text) are reliable.
"""
import re
from datetime import datetime

import numpy as np

import config as C
from common import haversine_m, num, photo_date_suspicious


def _exif(img):
    out = {"date": None, "lat": None, "lng": None}
    try:
        ex = img.getexif()
        dt = ex.get(306) or ex.get_ifd(0x8769).get(36867)
        if dt:
            out["date"] = datetime.strptime(str(dt)[:10], "%Y:%m:%d").strftime("%d%m%Y")
        gps = ex.get_ifd(0x8825)
        if gps and 2 in gps and 4 in gps:
            conv = lambda v, ref: (float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600) * (-1 if ref in "SW" else 1)
            out["lat"], out["lng"] = conv(gps[2], gps[1]), conv(gps[4], gps[3])
    except Exception:
        pass
    return out


def _stamp(img):
    """OCR the bottom strip (typical GPS-camera stamp) for lat/lng and date."""
    out = {"date": None, "lat": None, "lng": None}
    try:
        import pytesseract
        w, h = img.size
        txt = pytesseract.image_to_string(img.crop((0, int(h * 0.7), w, h)).convert("L"), lang="eng", config="--psm 6")
        m = re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})", txt)
        if m:
            out["date"] = f"{int(m[1]):02d}{int(m[2]):02d}{m[3]}"
        ll = re.findall(r"(\d{1,2}\.\d{3,})", txt)
        lo = re.findall(r"(\d{2}\.\d{3,})", txt)
        lat = next((float(x) for x in ll if C.INDIA_LAT_MIN <= float(x) <= C.INDIA_LAT_MAX), None)
        lng = next((float(x) for x in lo if C.INDIA_LNG_MIN <= float(x) <= C.INDIA_LNG_MAX), None)
        out["lat"], out["lng"] = lat, lng
    except Exception:
        pass
    return out


def _colour(img):
    """Return (green_frac, dry_frac, quality) from a downsized copy."""
    a = np.asarray(img.convert("RGB").resize((160, 120)), dtype=float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b
    green = float((exg > 20).mean())
    dry = float(((r > g * 0.95) & (r > b * 1.15) & (exg < 20) & (r > 90)).mean())  # yellow/brown vegetation & soil
    gray = a.mean(axis=2)
    lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    quality = "dark" if gray.mean() < 45 else "blurry" if lap < 2.5 else "good"
    return green, dry, quality


def _has_person(img):
    try:
        import cv2
        g = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
        casc = [cv2.CascadeClassifier(cv2.data.haarcascades + f) for f in ("haarcascade_frontalface_default.xml", "haarcascade_upperbody.xml")]
        s = 640 / max(g.shape)
        g = cv2.resize(g, None, fx=s, fy=s) if s < 1 else g
        return any(len(c.detectMultiScale(g, 1.1, 5, minSize=(40, 40))) for c in casc)
    except Exception:
        return False


def analyse(paths, row: dict) -> dict:
    from PIL import Image
    imgs = []
    for p in paths[:C.MAX_PHOTOS_PER_ROW]:
        try:
            imgs.append(Image.open(p))
            imgs[-1].load()
        except Exception:
            continue
    if not imgs:
        return {"photo_status": "Unavailable", "photo_error": "no readable photos"}
    lat, lng = num(row.get("latitude")), num(row.get("longitude"))
    flags, greens, drys, quals, dates, farmer = [], [], [], [], [], False
    far = 0
    for im in imgs:
        ex, st = _exif(im), _stamp(im)
        d = ex["date"] or st["date"]
        plat, plng = (ex["lat"], ex["lng"]) if ex["lat"] is not None else (st["lat"], st["lng"])
        if d:
            dates.append(d)
        if plat is not None and plng is not None and lat is not None and lng is not None:
            if haversine_m(lat, lng, plat, plng) > C.GPS_PHOTO_MAX_DISTANCE_M:
                far += 1
        g, dr, q = _colour(im)
        greens.append(g); drys.append(dr); quals.append(q)
        farmer = farmer or _has_person(im)
    if far:
        flags.append("GPS mismatch > 200m")
    photo_date = max(set(dates), key=dates.count) if dates else None
    if photo_date:
        import pandas as pd
        dd = datetime.strptime(photo_date, "%d%m%Y")
        if photo_date_suspicious(pd.Timestamp(dd), row):
            flags.append("Photo date outside survey period")
    g, dr = float(np.mean(greens)), float(np.mean(drys))
    field = "standing crop" if g >= 0.25 else "no crop" if g < 0.05 and dr < 0.25 else "cut & spread" if dr >= 0.25 else "standing crop"
    est = int(round(100 * dr / max(dr + g, 1e-6) / 5) * 5) if field != "no crop" else 100
    loss = num(row.get("crop_loss_pct"))
    if loss is not None and abs(est - loss) > C.PHOTO_LOSS_DIFF_FLAG_PCT:
        flags.append("Loss estimate differs from app (heuristic)")
    qual = max(set(quals), key=quals.count)
    return {"photo_status": "OK", "field_photo": field, "farmer_photo": farmer, "photo_loss": est, "photo_date": photo_date,
            "photo_quality": qual, "photo_flags": flags, "engine": "local-heuristic"}
