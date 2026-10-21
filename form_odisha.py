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


def _find_table_row(lines, headers_rx, y_search_from=0):
    """Find a header line matching all of headers_rx (in order, left-to-right), then the first non-empty line
    below it -> (col_x_bounds, row_text_or_None). Columns split at the midpoints between header word centres."""
    for i, ln in enumerate(lines):
        if ln["ycen"] < y_search_from:
            continue
        t = ln["text"].lower()
        if all(re.search(rx, t) for rx in headers_rx):
            # locate each header's x-centre by matching words greedily
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
            bounds = [-1e9] + [(centres[k] + centres[k + 1]) / 2 for k in range(len(centres) - 1)] + [1e9]
            # next non-empty line below (skip the header's own underline-only artifacts)
            for ln2 in lines[i + 1:i + 4]:
                if ln2["text"].strip():
                    row_words = ln2["words"]
                    cols = [[] for _ in headers_rx]
                    for w in row_words:
                        for k in range(len(headers_rx)):
                            if bounds[k] <= w["cx"] < bounds[k + 1]:
                                cols[k].append(w["text"])
                                break
                    return [" ".join(c).strip() or None for c in cols]
            return [None] * len(headers_rx)
    return None


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
    t1 = _find_table_row(lines, TABLE1_HDR)
    if t1:
        out.update(dict(zip(TABLE1_COLS, t1)))
    t2 = _find_table_row(lines, TABLE2_HDR)
    if t2:
        out.update(dict(zip(TABLE2_COLS, t2)))
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
