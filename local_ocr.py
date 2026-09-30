"""Free, offline form OCR (Tesseract eng+hin + rules + ink detection). No API key needed.

Reads a signed PMFBY form PDF/image and returns the same dict as pdf_qc.process():
form_area, form_loss, survey_date, surveyor/farmer/govt signed, form_status, confidence ...
Layout-agnostic: fields are found by label text (English/Hindi) next to a number, signatures by
ink present near a signature label. Tune LABELS / SIG_LABELS below against real forms.
"""
import re

import numpy as np

import config as C

TESS_LANG = "eng+hin"
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

LABELS = {
    "area": [r"affected\s*area", r"area\s*affected", r"प्रभावित\s*क्षेत्र", r"क्षति\s*क्षेत्र", r"प्रभावित\s*रकबा"],
    "loss": [r"crop\s*loss", r"loss\s*(?:of\s*crop)?\s*%", r"फसल\s*(?:की\s*)?(?:क्षति|हानि|नुकसान)", r"नुकसान"],
    "date": [r"date\s*of\s*(?:survey|inspection)", r"survey\s*date", r"सर्वे\s*(?:की\s*)?तारीख", r"सर्वेक्षण\s*(?:की\s*)?दिनांक", r"दिनांक"],
}
SIG_LABELS = {
    "surveyor": [r"surveyor", r"सर्वेयर", r"सर्वेक्षक", r"enumerator"],
    "farmer": [r"farmer", r"किसान", r"कृषक", r"अंगूठा", r"thumb"],
    "govt": [r"patwari", r"पटवारी", r"official", r"committee", r"अधिकारी", r"समिति", r"govt", r"revenue", r"कृषि\s*विभाग", r"tehsildar"],
}
NUM = r"(\d{1,3}(?:[.,]\d{1,2})?)"


def _ocr(img):
    import pytesseract
    d = pytesseract.image_to_data(img, lang=TESS_LANG, config="--psm 6", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if t:
            words.append({"t": t.translate(DEVANAGARI_DIGITS), "x": d["left"][i], "y": d["top"][i], "w": d["width"][i],
                          "h": d["height"][i], "c": float(d["conf"][i]), "line": (d["block_num"][i], d["par_num"][i], d["line_num"][i])})
    return words


def _lines(words):
    out = {}
    for w in words:
        out.setdefault(w["line"], []).append(w)
    return [sorted(v, key=lambda w: w["x"]) for _, v in sorted(out.items(), key=lambda kv: min(w["y"] for w in kv[1]))]


def _find_number(lines, patterns):
    """Number after a label on the same line (any line first); only then the next line's first number."""
    for allow_next in (False, True):
        for li, ln in enumerate(lines):
            text = " ".join(w["t"] for w in ln)
            for p in patterns:
                m = re.search(p, text, re.I)
                if not m:
                    continue
                n = re.search(NUM, text[m.end():])
                if not n and allow_next and li + 1 < len(lines):
                    n = re.search(NUM, " ".join(w["t"] for w in lines[li + 1]))
                if n:
                    v = float(n.group(1).replace(",", "."))
                    if 0 <= v <= 100:
                        conf = np.mean([w["c"] for w in ln if w["c"] >= 0] or [0]) / 100
                        return v, conf
    return None, 0.0


def _find_date(lines):
    for ln in lines:
        text = " ".join(w["t"] for w in ln)
        if any(re.search(p, text, re.I) for p in LABELS["date"]):
            m = re.search(r"(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{2,4})", text)
            if m:
                d, mo, y = m.groups()
                return f"{int(d):02d}{int(mo):02d}{y if len(y) == 4 else '20' + y}"
    return None


def _ink_above(gray, box, pad=6):
    """Fraction of dark pixels in the band just above/right of a label - a proxy for a signature."""
    x, y, w, h = box
    H, W = gray.shape
    regions = [gray[max(0, y - 3 * h):max(0, y - pad // 2), max(0, x - w):min(W, x + 2 * w)],   # above
               gray[y:y + h, min(W, x + w + 10):min(W, x + w + 6 * h * 6)]]                       # to the right
    best = 0.0
    for r in regions:
        if r.size:
            best = max(best, float((r < 140).mean()))
    return best


def _signatures(words, gray):
    found = {}
    for key, pats in SIG_LABELS.items():
        found[key] = False
        for w in words:
            if any(re.search(p, w["t"], re.I) for p in pats):
                # only count near a 'sign' word on the same line, to avoid body text
                same = [x["t"] for x in words if x["line"] == w["line"]]
                if not any(re.search(r"sign|हस्ताक्षर|अंगूठा|thumb|दस्तखत", t, re.I) for t in same):
                    continue
                if _ink_above(gray, (w["x"], w["y"], w["w"], w["h"])) > 0.03:
                    found[key] = True
                    break
    return found


def _render(path):
    import pymupdf
    from PIL import Image
    import io
    p = str(path)
    if p.lower().endswith(".pdf"):
        with pymupdf.open(p) as doc:
            return [Image.open(io.BytesIO(pg.get_pixmap(dpi=250).tobytes("png"))).convert("L") for pg in list(doc)[:C.PDF_MAX_PAGES]]
    return [__import__("PIL.Image", fromlist=["Image"]).open(p).convert("L")]


def extract(path, app_area=None, app_loss=None) -> dict:
    pages = _render(path)
    words_all, area, loss, date, sigs, confs = [], (None, 0), (None, 0), None, {}, []
    for img in pages:
        gray = np.array(img)
        words = _ocr(img)
        lines = _lines(words)
        words_all += words
        a, ac = _find_number(lines, LABELS["area"])
        l, lc = _find_number(lines, LABELS["loss"])
        area = (a, ac) if a is not None and area[0] is None else area
        loss = (l, lc) if l is not None and loss[0] is None else loss
        date = date or _find_date(lines)
        for k, v in _signatures(words, gray).items():
            sigs[k] = sigs.get(k, False) or v
        confs += [w["c"] for w in words if w["c"] >= 0]
    fa, fl = area[0], loss[0]
    ocr_conf = (np.mean(confs) / 100) if confs else 0.0
    field_conf = np.mean([c for v, c in (area, loss) if v is not None] or [0])
    conf = round(float(min(ocr_conf, field_conf) if fa is not None or fl is not None else ocr_conf * 0.5), 2)
    status = "incomplete" if fa is None or fl is None or not (sigs.get("surveyor") and sigs.get("farmer")) else "correct"
    res = {"pdf_status": "OK", "form_area": fa, "form_loss": fl, "survey_date": date,
           "surveyor_signed": bool(sigs.get("surveyor")), "farmer_signed": bool(sigs.get("farmer")),
           "govt_signed": bool(sigs.get("govt")), "form_remarks": None, "form_status": status,
           "pdf_confidence": conf, "manual_review": conf < C.LOW_CONFIDENCE_THRESHOLD, "engine": "local-ocr"}
    if fa is None or fl is None or app_area is None or app_loss is None:
        res["match"] = "NA"
    else:
        res["match"] = "Match" if abs(fa - app_area) <= C.AREA_MATCH_TOLERANCE_PCT and abs(fl - app_loss) <= C.LOSS_MATCH_TOLERANCE_PCT else "Mismatch"
    return res
