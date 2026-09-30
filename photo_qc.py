"""Geo-tagged photo analysis via Claude Vision (all photos in one call per row)."""
import base64
import re
from datetime import datetime

import httpx
import pandas as pd

import config as C
from common import photo_date_suspicious, Unavailable, call_claude, fetch, haversine_m, image_media_type, num, parse_dates

PROMPT = """These are geo-tagged field photos from a PMFBY crop loss survey for crop: {crop}, reported loss: {loss}%.

Analyze ALL photos and return a SINGLE JSON object:
{{
 "field_photo_type": one of EXACTLY: "no crop" (no crop in field: harvested, empty, wrong field), "cut & spread" (crop cut and spread on ground), "crop mismatch" (wrong crop type present), "standing crop" (crop standing in field),
 "farmer_photo_present": true if any photo shows a person (farmer),
 "loss_visible": true if crop damage is clearly visible,
 "estimated_loss_pct": estimated % crop loss visible (0,5,10...100) or null,
 "photo_date": date stamp visible on any photo as DDMMYYYY, or null,
 "photo_gps": {{"lat": number, "lng": number}} if GPS coords are visible, else null,
 "photo_quality": "good" | "blurry" | "dark" | "irrelevant",
 "loss_matches_reported": true if estimated_loss_pct is within {tol}% of the reported loss,
 "flags": list of short strings for any QC concerns
}}
Return JSON only."""

FIELD_TYPES = ("no crop", "cut & spread", "crop mismatch", "standing crop")


def split_urls(v) -> list[str]:
    v = str(v or "").replace("_x000D_", " ")  # Excel-escaped carriage returns
    return [u for u in re.split(r"[\s,;]+", v) if u.startswith("http")]


def _parse_date(s):
    try:
        return datetime.strptime(re.sub(r"\D", "", str(s)), "%d%m%Y")
    except ValueError:
        return None


async def process(row: dict, client, http: httpx.AsyncClient, tracker) -> dict:
    local = list(row.get("_local_photos") or [])[:C.MAX_PHOTOS_PER_ROW]
    urls = split_urls(row.get("media_urls"))[:max(0, C.MAX_PHOTOS_PER_ROW - len(local))]
    if not local and not urls:
        return {"photo_status": "Unavailable", "photo_error": "no urls"}
    content = []
    for u in [*local, *urls]:
        try:
            b = u.read_bytes() if not isinstance(u, str) else (await fetch(http, u, C.PHOTO_CACHE_DIR, C.PHOTO_TIMEOUT_SEC)).read_bytes()
        except (Unavailable, OSError, httpx.HTTPError, httpx.InvalidURL):
            continue
        mt = image_media_type(b)
        if mt and len(b) <= C.MAX_IMAGE_BYTES:
            content.append({"type": "image", "source": {"type": "base64", "media_type": mt,
                                                       "data": base64.b64encode(b).decode()}})
    if not content:
        return {"photo_status": "Unavailable", "photo_error": "no downloadable photos"}
    loss = num(row.get("crop_loss_pct"))
    content.append({"type": "text", "text": PROMPT.format(crop=row.get("crop_name") or "unknown",
                                                         loss="unknown" if loss is None else loss,
                                                         tol=C.PHOTO_LOSS_MATCH_PCT)})
    try:
        d = await call_claude(client, tracker, C.CLAUDE_MODEL_PHOTO, content)
    except Exception as e:
        return {"photo_status": "Error", "photo_error": f"{type(e).__name__}: {e}"[:200]}

    raw_flags = d.get("flags") or []
    flags = [str(f) for f in ([raw_flags] if isinstance(raw_flags, str) else raw_flags)]
    est = num(d.get("estimated_loss_pct"))
    gps = d.get("photo_gps")
    gps = gps if isinstance(gps, dict) else {}
    lat, lng = num(row.get("latitude")), num(row.get("longitude"))
    if num(gps.get("lat")) is not None and num(gps.get("lng")) is not None and lat is not None and lng is not None:
        if haversine_m(lat, lng, num(gps["lat"]), num(gps["lng"])) > C.GPS_PHOTO_MAX_DISTANCE_M:
            flags.append("GPS mismatch > 200m")
    pd_ = _parse_date(d.get("photo_date"))
    if pd_:
        if photo_date_suspicious(pd.Timestamp(pd_), row):
            flags.append("Photo date outside survey period")
    if est is not None and loss is not None and abs(est - loss) > C.PHOTO_LOSS_DIFF_FLAG_PCT:
        flags.append("Loss estimate differs from app")
    ft = d.get("field_photo_type")
    return {"photo_status": "OK", "field_photo": ft if ft in FIELD_TYPES else None,
            "farmer_photo": bool(d.get("farmer_photo_present")), "photo_loss": est,
            "photo_date": d.get("photo_date"), "photo_quality": d.get("photo_quality"),
            "photo_flags": flags}
