"""PDF form OCR via Claude Vision."""
import base64

import httpx

import config as C
from common import Unavailable, call_claude, fetch, num

PROMPT = """This is a PMFBY crop loss assessment form. The form may be in Hindi, English, or a regional language. Extract ONLY these fields:

1. affected_area_percent: percentage of affected/damaged area (labels like: affected area%, Affected Area%, क्षति क्षेत्र). Number 0-100, or null.
2. crop_loss_percent: percentage of crop loss. Number 0-100, or null.
3. survey_date: date the survey was conducted, DDMMYYYY, or null.
4. surveyor_signed: true/false - surveyor signature present.
5. farmer_signed: true/false - farmer signature/thumb impression present.
6. govt_signed: true/false - government official / committee signature present.
7. form_remarks: any handwritten or printed remarks/notes on the form, or null.
8. form_status: 'correct' (all key fields filled, signatures present), 'incomplete' (a key field is blank), or 'overwrite' (values appear corrected/overwritten/tampered).
9. confidence: your confidence in the extraction, 0.0-1.0.

Return a single JSON object with exactly these keys and nothing else."""


def _render(path) -> list[bytes]:
    import pymupdf
    imgs = []
    with pymupdf.open(path) as doc:
        for page in list(doc)[:C.PDF_MAX_PAGES]:
            imgs.append(page.get_pixmap(dpi=C.PDF_RENDER_DPI).tobytes("png"))
    return imgs


async def process(row: dict, client, http: httpx.AsyncClient, tracker) -> dict:
    url = str(row.get("pdf_url") or "").replace("_x000D_", "").strip()
    local = row.get("_local_pdfs") or []
    if not local and not url.startswith("http"):
        return {"pdf_status": "Unavailable", "pdf_error": "no url"}
    try:
        path = local[0] if local else await fetch(http, url, C.PDF_CACHE_DIR, C.PDF_TIMEOUT_SEC)
        pages = _render(path)
    except Unavailable as e:
        return {"pdf_status": "Unavailable", "pdf_error": str(e)}
    except Exception as e:
        return {"pdf_status": "Unavailable", "pdf_error": f"render: {type(e).__name__}"}
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                           "data": base64.b64encode(p).decode()}} for p in pages]
    content.append({"type": "text", "text": PROMPT})
    try:
        d = await call_claude(client, tracker, C.CLAUDE_MODEL_PDF, content)
    except Exception as e:
        return {"pdf_status": "Error", "pdf_error": f"{type(e).__name__}: {e}"[:200]}
    conf = min(max(num(d.get("confidence")) or 0.0, 0.0), 1.0)
    fa, fl = num(d.get("affected_area_percent")), num(d.get("crop_loss_percent"))
    res = {"pdf_status": "OK", "form_area": fa, "form_loss": fl, "survey_date": d.get("survey_date"),
           "surveyor_signed": bool(d.get("surveyor_signed")), "farmer_signed": bool(d.get("farmer_signed")),
           "govt_signed": bool(d.get("govt_signed")), "form_remarks": d.get("form_remarks"),
           "form_status": d.get("form_status") if d.get("form_status") in ("correct", "incomplete", "overwrite") else "incomplete",
           "pdf_confidence": conf, "manual_review": conf < C.LOW_CONFIDENCE_THRESHOLD}
    a, l = num(row.get("affected_area_pct")), num(row.get("crop_loss_pct"))
    if fa is None or fl is None or a is None or l is None:
        res["match"] = "NA"
    else:
        ok = abs(fa - a) <= C.AREA_MATCH_TOLERANCE_PCT and abs(fl - l) <= C.LOSS_MATCH_TOLERANCE_PCT
        res["match"] = "Match" if ok else "Mismatch"
    return res
