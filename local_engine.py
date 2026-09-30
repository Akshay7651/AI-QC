"""Adapters that give the free local OCR/photo engines the same signature as pdf_qc/photo_qc.process."""
import asyncio

import config as C
from common import Unavailable, fetch, num
import local_ocr
import photo_local
from photo_qc import split_urls


async def process_pdf(row, client, http, tracker):
    local = row.get("_local_pdfs") or []
    url = str(row.get("pdf_url") or "").replace("_x000D_", "").strip()
    if not local and not url.startswith("http"):
        return {"pdf_status": "Unavailable", "pdf_error": "no url"}
    try:
        path = local[0] if local else await fetch(http, url, C.PDF_CACHE_DIR, C.PDF_TIMEOUT_SEC)
        return await asyncio.to_thread(local_ocr.extract, path, num(row.get("affected_area_pct")), num(row.get("crop_loss_pct")))
    except Unavailable as e:
        return {"pdf_status": "Unavailable", "pdf_error": str(e)}
    except Exception as e:
        return {"pdf_status": "Error", "pdf_error": f"{type(e).__name__}: {e}"[:200]}


async def process_photo(row, client, http, tracker):
    paths = list(row.get("_local_photos") or [])[:C.MAX_PHOTOS_PER_ROW]
    for u in split_urls(row.get("media_urls"))[:max(0, C.MAX_PHOTOS_PER_ROW - len(paths))]:
        try:
            paths.append(await fetch(http, u, C.PHOTO_CACHE_DIR, C.PHOTO_TIMEOUT_SEC))
        except Unavailable:
            pass
    if not paths:
        return {"photo_status": "Unavailable", "photo_error": "no photos"}
    try:
        return await asyncio.to_thread(photo_local.analyse, paths, row)
    except Exception as e:
        return {"photo_status": "Error", "photo_error": f"{type(e).__name__}: {e}"[:200]}
