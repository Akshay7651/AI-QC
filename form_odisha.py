"""Reader for the Odisha 'Crop Loss Joint Survey Form' (Generali Central Insurance, PMFBY).

Unlike Proforma-3 (form_p3.py / form_reader.py) this form is printed in English with free-text "label: value"
lines rather than a strict Hindi ruled grid, so it is read with label-anchored OCR instead of column geometry:
full-page Tesseract OCR -> group words into lines -> for each known label, regex the text that follows it on
the same line (stopping at the next known label, if the line holds two fields) up to the next ruled table.

Deskew/orientation reuses form_p3.normalise() (ruled-line based, ok here too - the form has several ruled
tables) since the forms are camera photos, not flatbed scans (visible tilt/skew per sample).
"""
import re

import cv2
import numpy as np

import form_p3 as P

HEADER_RE = re.compile(r"crop\s*loss\s*joint\s*survey\s*form", re.I)

# (field key, label regexes to match at the START of the remaining line text, stop regex for a second field
# sharing the same line). Order matters only for same-line disambiguation.
_FIELDS = [
    ("s_no", r"s\s*no\.?"),
    ("gci_reference_no", r"1\)?\s*gci\s*reference\s*no\.?"),
    ("date", r"date\s*[:.\-]*"),
    ("lot_no", r"lot\s*no\.?"),
    ("crop_name", r"2\)?\s*crop\s*name"),
    ("policy_application_no", r"policy\s*no\.?\s*/?\s*application\s*no\.?"),
    ("farmer_name", r"3\)?\s*farmer'?s?\s*name\s*[:.\-]*"),
    ("crop_season_year", r"crop\s*season\s*&?\s*year"),
    ("mobile_no", r"mobile\s*no\.?\s*[:.\-]*"),
    ("state", r"state\s*[_:.\-]*"),
    ("district", r"district\s*[_:.\-]*"),
    ("tehsil", r"tehsil\s*[_:.\-]*"),
    ("patwar_name", r"patwar\s*name"),
    ("village", r"village\s*[_:.\-]*"),
    ("bank_account", r"6\)?\s*bank\s*account\s*number\s*\(?kcc.?other\)?"),
    ("ifsc", r"ifsc\s*code\s*[:.\-]*"),
    ("bank_name", r"name\s*of\s*bank"),
    ("branch", r"branch\s*[_:.\-]*"),
    ("event_occurred_date", r"8\)?\s*event\s*occurred\s*date"),
    ("intimation_date", r"date\s*and\s*time\s*of\s*intimations?"),
    ("age_of_crop", r"12\)?\s*age\s*of\s*crop\s*\(?in\s*days\)?"),
    ("crop_condition", r"14\)?\s*crop\s*condition\s*at\s*the\s*time\s*of\s*survey"),
    ("surveyor_remarks", r"17\)?\s*surveyor\s*remarks\s*[:.\-]*"),
]
_FIELD_RE = [(k, re.compile(r"^\s*" + pat, re.I)) for k, pat in _FIELDS]

# the loss-detail table header row (two tables use this shape): label -> column index
TABLE1_COLS = ["survey_no", "actual_crop", "total_area_hect", "insured_area_hect", "sowing_date"]
TABLE1_HDR = [r"survey\s*no", r"actual\s*crop", r"total\s*area", r"insured\s*area", r"sowing\s*date"]
TABLE2_COLS = ["affected_area_pct", "loss_pct", "farmer_aware"]
TABLE2_HDR = [r"affected\s*area", r"loss\s*%?", r"farmer\s*is\s*aware"]


def _ocr_data(gray):
    import pytesseract
    d = pytesseract.image_to_data(gray, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT)
    words = []
    for i in range(len(d["text"])):
        t = d["text"][i].strip()
        if not t or float(d.get("conf", [0])[i]) < 0:
            continue
        if not t:
            continue
        words.append({"text": t, "x": d["left"][i], "y": d["top"][i], "w": d["width"][i], "h": d["height"][i],
                      "cx": d["left"][i] + d["width"][i] / 2, "cy": d["top"][i] + d["height"][i] / 2})
    return words


def _group_lines(words, tol=0.6):
    """Group words into text lines by vertical overlap; each line sorted left-to-right."""
    if not words:
        return []
    hs = sorted(w["h"] for w in words)
    med_h = hs[len(hs) // 2] or 12
    ws = sorted(words, key=lambda w: w["cy"])
    lines, cur, cur_y = [], [], None
    for w in ws:
        if cur and abs(w["cy"] - cur_y) > tol * med_h:
            lines.append(cur)
            cur = []
        cur.append(w)
        cur_y = sum(x["cy"] for x in cur) / len(cur)
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        ln = sorted(ln, key=lambda w: w["x"])
        text = " ".join(w["text"] for w in ln)
        y0 = min(w["y"] for w in ln); y1 = max(w["y"] + w["h"] for w in ln)
        out.append({"words": ln, "text": text, "y0": y0, "y1": y1, "ycen": (y0 + y1) / 2})
    return out


def is_odisha_form(words_or_text) -> bool:
    text = words_or_text if isinstance(words_or_text, str) else " ".join(w["text"] for w in words_or_text)
    t = text.lower()
    return bool(HEADER_RE.search(t)) or ("generali" in t and "crop loss" in t) or \
        ("survey no" in t and "actual crop" in t and "insured area" in t)


def _clean_value(s):
    s = re.sub(r"^[\s:_\-.|]+", "", s).strip()
    s = re.sub(r"[_]{2,}", "", s).strip(" _-:.")
    return s or None


def _extract_header_fields(lines):
    out = {}
    for ln in lines:
        text = ln["text"]
        hits = []
        for key, rx in _FIELD_RE:
            m = rx.search(text)
            if m:
                hits.append((m.start(), m.end(), key))
        if not hits:
            continue
        hits.sort()
        for i, (s, e, key) in enumerate(hits):
            end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
            val = _clean_value(text[e:end])
            if val and key not in out:
                out[key] = val
    return out


def _row_band(lines, after_y1, next_rx, max_gap):
    """Row band (y0,y1) starting right after `after_y1`: everything up to the line matching `next_rx` (a known
    anchor further down the form), capped at `max_gap` px. Spans every OCR line-group in that range (handwriting
    with tall ascenders/descenders often splits into more than one line-group, so picking a single "next line"
    is unreliable - the union of everything in the band is not)."""
    y1 = after_y1 + max_gap
    for ln in lines:
        if ln["ycen"] > after_y1 and next_rx and re.search(next_rx, ln["text"], re.I):
            y1 = min(y1, ln["y0"])
            break
    band = [ln for ln in lines if after_y1 < ln["ycen"] <= y1]
    if not band:
        return after_y1, after_y1 + max_gap
    return min(ln["y0"] for ln in band), max(ln["y1"] for ln in band)


def _find_table_box(lines, headers_rx, y_search_from=0, next_rx=None, max_gap=140):
    """Find a header line matching all of headers_rx (in order, left-to-right) -> dict(col_bounds: [x0..xN] in
    image px, hdr_y1, row_y0, row_y1). The row band is the union of every OCR line between the header and the
    next known section (see _row_band) rather than a single "next line", which handwriting easily splits."""
    for i, ln in enumerate(lines):
        if ln["ycen"] < y_search_from:
            continue
        t = ln["text"].lower()
        if not all(re.search(rx, t) for rx in headers_rx):
            continue
        centres = []
        words = ln["words"]
        wi = 0
        for rx in headers_rx:
            pat = re.compile(rx, re.I)
            found = None
            for j in range(wi, len(words)):
                joined = " ".join(w["text"] for w in words[wi:j + 1])
                if pat.search(joined.lower()):
                    found = words[wi:j + 1]
                    wi = j + 1
                    break
            if found:
                centres.append(sum(w["cx"] for w in found) / len(found))
        if len(centres) < len(headers_rx):
            continue
        W = max(w["x"] + w["w"] for ln3 in lines for w in ln3["words"])
        bounds = [0.0] + [(centres[k] + centres[k + 1]) / 2 for k in range(len(centres) - 1)] + [float(W)]
        row_y0, row_y1 = _row_band(lines, ln["y1"], next_rx, max_gap)
        return {"col_bounds": bounds, "hdr_y1": ln["y1"], "row_y0": row_y0, "row_y1": row_y1}
    return None


def _find_table_box_by_anchor(lines, anchor_rx, next_rx, n_cols, x0, x1, max_gap=160):
    """Fallback when a table's own column headers don't OCR reliably (thin/bordered header text): anchor off a
    nearby line that DOES read well (e.g. the numbered section title above the table), skip the printed column-
    header line right under it, then take the row band below THAT as the handwritten data row, split into
    `n_cols` equal-width columns spanning (x0,x1) - valid because this is a fixed printed template (same column
    layout on every form), unlike Proforma-3 where geometry is detected per-form."""
    for i, ln in enumerate(lines):
        if not re.search(anchor_rx, ln["text"], re.I):
            continue
        hdr_row_y1 = ln["y1"]
        for ln2 in lines[i + 1:i + 3]:
            if ln2["text"].strip():
                hdr_row_y1 = ln2["y1"]
                break
        row_y0, row_y1 = _row_band(lines, hdr_row_y1, next_rx, max_gap)
        bounds = [x0 + (x1 - x0) * k / n_cols for k in range(n_cols + 1)]
        return {"col_bounds": bounds, "hdr_y1": ln["y1"], "row_y0": row_y0, "row_y1": row_y1}
    return None


def _find_table_row(lines, headers_rx, y_search_from=0):
    """Text-only convenience wrapper over _find_table_box (OCR line text per column, no re-cropping)."""
    box = _find_table_box(lines, headers_rx, y_search_from)
    if box is None:
        return None
    bounds = box["col_bounds"]
    for ln in lines:
        if box["row_y0"] - 2 <= ln["ycen"] <= box["row_y1"] + 2:
            cols = [[] for _ in range(len(bounds) - 1)]
            for w in ln["words"]:
                for k in range(len(bounds) - 1):
                    if bounds[k] <= w["cx"] < bounds[k + 1]:
                        cols[k].append(w["text"])
                        break
            return [" ".join(c).strip() or None for c in cols]
    return [None] * (len(bounds) - 1)


def _ocr_cell(gray, box, digits_only=False, pad=4):
    """Crop (x0,y0,x1,y1) from `gray`, upscale + binarize, OCR one line. digits_only restricts to a numeric
    whitelist (area%, loss%, survey no, dates are all short digit/slash/percent tokens - this reads far more
    reliably than free-text OCR on handwriting)."""
    import pytesseract
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(gray.shape[1], x1 + pad), min(gray.shape[0], y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None, 0.0
    c = gray[y0:y1, x0:x1]
    if c.size == 0:
        return None, 0.0
    scale = max(1.0, 220.0 / c.shape[0])
    c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)
    _, b = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = float((b < 128).mean())
    if digits_only:
        cfg = "--psm 7 -c tessedit_char_whitelist=0123456789./%-"
    else:
        cfg = "--psm 7"
    txt = pytesseract.image_to_string(b, lang="eng", config=cfg).strip()
    return (txt or None), ink


def _read_table_cells(gray, box, keys, digit_cols):
    """Re-OCR each column of a table row at full resolution (better than the small-page OCR pass used for
    header detection). digit_cols: set of indices to read with the numeric whitelist."""
    out = {}
    bounds = box["col_bounds"]
    for k, key in enumerate(keys):
        cell = (bounds[k], box["row_y0"], bounds[k + 1], box["row_y1"])
        txt, ink = _ocr_cell(gray, cell, digits_only=(k in digit_cols))
        out[key] = txt
        out[key + "_ink"] = round(ink, 4)
    return out


def _read_signatures(lines, gray):
    """Participants table (section 18): SNo | Post | Name | Signature. Ink-only check per row - no OCR needed,
    so this is reliable regardless of handwriting quality."""
    box = _find_table_box(lines, [r"s\s*no", r"post", r"name", r"signature"])
    if box is None:
        return {}
    bounds = box["col_bounds"]
    sig_x0, sig_x1 = bounds[-2], bounds[-1]
    rows = [ln for ln in lines if ln["ycen"] > box["hdr_y1"] and ln["ycen"] < box["hdr_y1"] + 5 * (box["row_y1"] - box["row_y0"] + 20)]
    out = {}
    for i, ln in enumerate(rows[:5]):
        cell = (sig_x0, ln["y0"], sig_x1, ln["y1"])
        _, ink = _ocr_cell(gray, cell)
        out[f"participant_{i + 1}_signed"] = bool(ink >= 0.03)
    return out


def read_odisha(rgb, debug=False):
    """rgb: upright-ish page (orientation not yet corrected). -> dict of extracted fields, or
    {'is_odisha_form': False} if this isn't the Odisha template."""
    r, info = P.normalise(rgb)
    g = cv2.cvtColor(P._resize_w(r, 1600), cv2.COLOR_RGB2GRAY)
    words = _ocr_data(g)
    if not is_odisha_form(words):
        return {"is_odisha_form": False}
    lines = _group_lines(words)
    out = {"is_odisha_form": True, "_quality": {"n_words": len(words)}}
    out.update(_extract_header_fields(lines))
    box1 = _find_table_box(lines, TABLE1_HDR, next_rx=r"7\)\s*reason\s*of\s*crop\s*loss")
    if box1:
        out.update(_read_table_cells(g, box1, TABLE1_COLS, digit_cols={0, 2, 3, 4}))
    box2 = _find_table_box(lines, TABLE2_HDR, next_rx=r"16\)\s*loss\s*survey\s*mandatory")
    if box2 is None and box1 is not None:
        # the table-2 header text ("Affected Area | Loss% | The farmer is aware...") often fails to OCR as a
        # clean line (thin/bordered text) - fall back to the "15) Total Information..." anchor above it, using
        # table 1's column width (same printed template, both tables span the same left-right extent)
        box2 = _find_table_box_by_anchor(lines, r"15\)\s*total\s*information", r"16\)\s*loss\s*survey\s*mandatory",
                                         3, box1["col_bounds"][0], box1["col_bounds"][-1])
    if box2:
        out.update(_read_table_cells(g, box2, TABLE2_COLS, digit_cols={0, 1}))
    out.update(_read_signatures(lines, g))
    # surveyor remarks often spans to the next line too (free handwriting, no box) - append one more line
    for i, ln in enumerate(lines):
        if re.search(r"17\)?\s*surveyor\s*remarks", ln["text"], re.I) and "surveyor_remarks" in out:
            nxt = lines[i + 1]["text"].strip() if i + 1 < len(lines) else ""
            if nxt and not re.search(r"^\s*18\)", nxt, re.I) and len(nxt) < 60:
                out["surveyor_remarks"] = (out["surveyor_remarks"] + " " + nxt).strip()
            break
    if debug:
        out["_lines"] = [ln["text"] for ln in lines]
    return out
