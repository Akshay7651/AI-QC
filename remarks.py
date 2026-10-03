"""Row-level QC reasoning: turns what the form reader / photo analyst / GPS / data / risk engines found into
  * a precise, prioritised, human-readable remark  ('Any Other Remarks')
  * a QC Verdict  (OK / Partially OK / Review / Manual QC Required)
  * AI_Confidence (High / Medium / Low), AI_Flags (short tags) and counter keys for the live dashboard.

Nothing here invents values: a field whose confidence is low is reported as 'not readable'.

VERDICT RULES (first matching rule wins, top to bottom)
  Reject-evidence  the evidence cannot support the claim:
                     - no signed-form link in the record, or no photo link, or
                     - every uploaded photo is a picture of the paper form (no field photograph), or
                     - the uploaded 'form' is not a Proforma-3 document.
  Manual-check     the machine cannot decide, a person must look:
                     - form/photos could not be downloaded or opened (link present, file unavailable),
                     - the form is unreadable / key values are low confidence,
                     - overall confidence below 0.6.
  Review           readable evidence with problems a reviewer should confirm: form vs app mismatch,
                   inconsistent row/total, PO ID mismatch, overwriting, missing signature(s),
                   officer stamp only, duplicate photos, photo GPS > 200 m from the app point,
                   photo date outside the survey window, rotated photos, field state vs reported loss,
                   GPS cluster flags, data-QC flags, risk score >= 60.
  OK               none of the above.

CONFIDENCE  Low: evidence missing/failed or conf < 0.6;  Medium: any Review flag, conf < 0.8 or one of form/photos absent;
            High: readable form and photos, no flags.
"""
from __future__ import annotations

import re
from datetime import datetime

import config as C

VERDICTS = ("OK", "Partially OK", "Review", "Manual QC Required")
LOW_FIELD_CONF = 0.5
RISK_REVIEW = 60
COVERED_DATA_FLAGS = ("Missing: signed form link", "Missing: photo links")

# counters shown on the live dashboard: counter key -> flag text that increments it
COUNTER_OF = {
    "Form not found": "form_not_found",
    "Form link missing": "missing_form_link",
    "Photo link missing": "missing_photo_link",
    "Photo is form image": "photo_is_form",
    "No crop in photo": "no_crop_in_photo",
    "Crop mismatch": "crop_mismatch",
    "Flooding seen": "flooded",
    "Photo not of the field": "photo_not_field",
    "Same location": "same_location",
    "Photo GPS mismatch": "gps_mismatch",
    "Signature missing": "signature_missing",
    "Form vs app mismatch": "mismatch",
    "Form overwrite": "overwrite",
    "Duplicate photos": "duplicate_photos",
    "Form not readable": "form_unreadable",
    "Not Proforma-3": "not_proforma3",
    "GPS cluster": "gps_cluster",
    "PO ID mismatch": "po_id_mismatch",
}


# ---------------------------------------------------------------- small helpers
def _blank(v):
    return v is None or (isinstance(v, float) and v != v) or str(v).strip().lower() in ("", "nan", "none", "nat")


def _num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _fmt(v):
    f = _num(v)
    if f is None:
        return "?"
    return str(int(f)) if f == int(f) else f"{f:g}"


def _yn(v):
    return "Yes" if v else "No"


def _parse_date(v):
    """DDMMYYYY / DD-MM-YYYY / ISO -> datetime or None."""
    if _blank(v):
        return None
    s = str(v).strip()
    for fmt, ln in (("%d%m%Y", 8), ("%d-%m-%Y", 10), ("%d/%m/%Y", 10), ("%Y-%m-%d", 10), ("%d.%m.%Y", 10)):
        try:
            return datetime.strptime(s[:ln], fmt)
        except ValueError:
            continue
    return None


def _fconf(form, *names):
    """Field confidence from form['field_conf'] (None if unknown)."""
    fc = (form or {}).get("field_conf") or {}
    for n in names:
        v = _num(fc.get(n))
        if v is not None:
            return v
    return None


def _readable(form, value, *names):
    """True if value is present and its confidence is not low."""
    if value is None:
        return False
    c = _fconf(form, *names)
    if c is None:  # no per-field confidence: fall back to the overall form confidence
        oc = _num((form or {}).get("confidence"))
        return oc is None or oc >= C.LOW_CONFIDENCE_THRESHOLD
    return c >= LOW_FIELD_CONF


# ---------------------------------------------------------------- core
# ---------------------------------------------------------------- plain-language remark (for non-technical readers)
_REASON = {
    "Form link missing": "the signed form link is missing",
    "Photo link missing": "the photo link is missing",
    "Form not found": "the signed form could not be downloaded",
    "Photos not found": "the photos could not be downloaded",
    "Not Proforma-3": "the uploaded paper is not the Proforma-3 form",
    "Form vs app mismatch": "the values on the form are different from the app",
    "Form row/total inconsistent": "the row and total on the form do not agree",
    "Form overwrite": "there is overwriting / cutting on the form",
    "Form not readable": "the handwriting (area / loss) on the form is not clear enough for the AI to read",
    "Low confidence - manual review": "the form is not clear enough for the AI to read",
    "Photo GPS mismatch": "the photo was taken far from the app location",
    "Photo date outside survey period": "the photo date is outside the survey period",
    "Same location": "several surveys were done at the same spot",
    "Same app entry at same spot": "the surveys at the same spot all have the same app entry (possible copy)",
    "GPS cluster": "many surveys by the same surveyor at one spot",
    "Field state vs reported loss": "the field in the photo does not match the reported loss",
    "Crop mismatch": "the crop in the photo is different from the declared crop",
    "No crop in photo": "no crop is visible in the photo but the reported loss is low",
    "Photo not of the field": "the photos do not show the field",
    "Officer stamp only": "the officer block has only a stamp, no signature",
}


_SHORT_TAG = {
    "Form link missing": "Form Link Missing",
    "Photo link missing": "Photo Link Missing",
    "Form not found": "Form Not Found",
    "Photos not found": "Photos Not Found",
    "Not Proforma-3": "Not Proforma-3 (Wrong Document)",
    "Form not readable": "Form Not Readable (Handwriting Unclear)",
    "Low confidence - manual review": "Low Confidence (Verify Manually)",
    "Signature missing": "Signature Missing",
    "Photo is form image": "Form Image Uploaded Instead of Field Photo",
    "Photo not of the field": "Wrong Photo (Not a Field Photo)",
    "Field state vs reported loss": "Photo vs Reported Loss Mismatch",
    "Crop mismatch": "Crop in Photo Differs from Declared Crop",
    "Form vs app mismatch": "Form & App Values Mismatch",
    "Flooding seen": "Flooding Visible in Photo",
    "Photo GPS mismatch": "Photo Location Differs from App GPS",
    "Photo date outside survey period": "Photo Date Outside Survey Period",
    "Same location": "Same Location",
    "Same app entry at same spot": "Same Location (Possible Copied Entry)",
    "GPS cluster": "Multiple Surveys at Same Spot",
    "Overwriting": "Overwriting Detected on Form",
}

_INFO_ONLY = {"Duplicate photos", "Photos rotated"}     # informational: shown in the detailed remark, not in the verdict line


def _plain(row, form, photos, gps, flags, verdict, ev):
    tags = []
    same_spot = any(f in flags for f in ("Same location", "GPS cluster", "Same app entry at same spot"))
    for f in flags:
        if f in _INFO_ONLY:
            continue
        if f in ("Same location", "GPS cluster"):          # one place-related tag instead of two that say the same thing
            if "Same app entry at same spot" not in flags:
                tags.append("Multiple Surveys at Same Spot")
            continue
        if f == "Data: Possible duplicate field" and same_spot:
            continue                                        # the same-spot tag already covers it
        if f == "Data: Possible duplicate field":
            tags.append("Same Land Record (Khasra) in Another Docket")
            continue
        if f.startswith("Data: "):
            tags.append("Data Issue (" + f[6:] + ")")
        elif f.startswith("High risk score"):
            tags.append("High Risk Score")
        elif f in _SHORT_TAG:
            tags.append(_SHORT_TAG[f])
        elif f not in ("Internal error",):
            tags.append(f)
    fa, fl, m = ev.get("form_area"), ev.get("form_loss"), ev.get("match")
    if m == "Mismatch" and "Form & App Values Mismatch" not in tags:
        tags.append("Form & App Values Mismatch")
    tags = list(dict.fromkeys(tags))
    if not tags:
        return verdict
    return verdict + " - " + ", ".join(tags)


def evaluate(row, form=None, photos=None, gps=None, data_flags="", risk=None):
    """Return dict(remark, verdict, confidence, flags, counters, match, form_area, form_loss, form_source)."""
    row = row or {}
    gps = gps or {}
    risk = risk or {}
    flags: list[str] = []          # short tags (AI_Flags)
    severity = {"reject": [], "manual": [], "review": [], "partial": []}
    parts_form: list[str] = []
    parts_photo: list[str] = []
    parts_gps: list[str] = []
    headline: list[str] = []

    def flag(text, level, headline_text=None):
        flags.append(text)
        if level:  # level None = informational tag only (no effect on the verdict)
            severity[level].append(text)
        if headline_text:
            headline.append(headline_text)

    app_area, app_loss = _num(row.get("affected_area_pct")), _num(row.get("crop_loss_pct"))
    form_url_blank = _blank(row.get("pdf_url"))
    photo_url_blank = _blank(row.get("media_urls"))
    match, f_area, f_loss, source = "NA", None, None, None

    # ------------------------------------------------------------ FORM
    fstate = (form or {}).get("_state") if form else None
    if fstate is None:
        fstate = "no_link" if (form is None and form_url_blank) else ("not_found" if form is None or form.get("_error") else "ok")
    if fstate == "skipped":
        pass
    elif fstate == "no_link":
        flag("Form link missing", "reject", "signed form link missing in the record")
        parts_form.append("No signed-form link in the record, so the form could not be checked.")
    elif fstate in ("not_found", "error"):
        err = (form or {}).get("_error")
        flag("Form not found", "manual", "signed form could not be retrieved")
        parts_form.append("Signed form could not be retrieved" + (f" ({err})" if err else "") + " - check the link / download manually.")
    else:
        conf = _num(form.get("confidence"))
        conf_low = conf is not None and conf < C.LOW_CONFIDENCE_THRESHOLD
        if form.get("is_proforma3") is False:
            flag("Not Proforma-3", "reject", "uploaded form is not a Proforma-3 document")
            parts_form.append("Uploaded form is not a Proforma-3 document" + (f" (image quality: {form['quality']})" if form.get("quality") else "") + ".")
        else:
            q = form.get("quality")
            if q and q != "good":
                parts_form.append(f"Form image quality: {q}.")
            # identity
            fn, fn_ok = form.get("form_no"), _readable(form, form.get("form_no"), "form_no")
            s = f"Form No {fn}" if fn and fn_ok else ("Form No not applicable (Rajasthan form)" if form.get("template") == "rajasthan" else "Form No not readable")
            po = form.get("po_id_matches")
            if po is True:
                s += "; PO ID matches docket"
            elif po is False:
                flag("PO ID mismatch", "partial", "PO ID on the form differs from the docket")
                shown = form.get("po_id") if _readable(form, form.get("po_id"), "po_id") else None
                s += f"; PO ID {shown} does NOT match docket" if shown else "; PO ID does NOT match docket"
            # (PO ID reading is skipped by design: say nothing when it is unknown)
            parts_form.append(s + ".")
            # values
            fa, fl = _num(form.get("form_area")), _num(form.get("form_loss"))
            ra, rl = _num(form.get("row_area")), _num(form.get("row_loss"))
            fa_ok, fl_ok = _readable(form, fa, "area", "form_area"), _readable(form, fl, "loss", "form_loss")
            ra_ok, rl_ok = _readable(form, ra, "row_area", "area"), _readable(form, rl, "row_loss", "loss")
            total_blank = bool(form.get("total_row_blank"))
            if fa is not None and fl is not None and fa_ok and fl_ok:
                f_area, f_loss, source = fa, fl, "total row"
            elif ra is not None and rl is not None and ra_ok and rl_ok:
                f_area, f_loss, source = ra, rl, "table row"
            elif fa is not None and fa_ok and rl is not None and rl_ok:
                f_area, f_loss, source = fa, rl, "mixed (total area, row loss)"
            elif ra is not None and ra_ok and fl is not None and fl_ok:
                f_area, f_loss, source = ra, fl, "mixed (row area, total loss)"
            # only one of the two values readable: do NOT invent the other one (it used to be set to 0) - report 'not readable'
            raj = form.get("template") == "rajasthan"
            pairs = form.get("raj_pairs")
            if f_area is None and raj and pairs and app_area is not None and app_loss is not None:
                hit = [(a_, l_) for a_, l_ in pairs if abs(a_ - app_area) <= C.AREA_MATCH_TOLERANCE_PCT and abs(l_ - app_loss) <= C.LOSS_MATCH_TOLERANCE_PCT]
                if hit:
                    f_area, f_loss, source = hit[0][0], hit[0][1], "one of several rows"
                elif form.get("raj_all_read"):
                    parts_form.append(f"None of the {len(pairs)} filled rows on the form shows the app values {_fmt(app_area)}% / {_fmt(app_loss)}%.")
                    flag("Form vs app mismatch", "review", f"no row on the form equals the app values {_fmt(app_area)}/{_fmt(app_loss)}")
                    match = "Mismatch"
            if f_area is None:
                parts_form.append("Affected area % / loss % not readable on the form.")
                if not conf_low:
                    flag("Form not readable", "manual")
            else:
                if source == "total row":
                    s = f"Form total affected area {_fmt(f_area)}% / loss {_fmt(f_loss)}%"
                elif source.startswith("mixed"):
                    s = f"Form area {_fmt(f_area)}% / loss {_fmt(f_loss)}% ({source})"
                elif source == "one of several rows":
                    s = f"One of the {len(pairs)} rows on the form shows affected area {_fmt(f_area)}% / loss {_fmt(f_loss)}%"
                elif raj:
                    s = f"Form row shows affected area {_fmt(f_area)}% / loss {_fmt(f_loss)}%"
                else:
                    s = f"Form total row not usable; table row shows affected area {_fmt(f_area)}% / loss {_fmt(f_loss)}%"
                if app_area is None or app_loss is None:
                    match = "NA"
                    s += " (app values missing)"
                else:
                    ok = (abs(f_area - app_area) <= C.AREA_MATCH_TOLERANCE_PCT and abs(f_loss - app_loss) <= C.LOSS_MATCH_TOLERANCE_PCT)
                    match = "Match" if ok else "Mismatch"
                    if ok:
                        s += f" = app (Match)"
                    else:
                        s += f" vs app {_fmt(app_area)}% / {_fmt(app_loss)}% (Mismatch)"
                        flag("Form vs app mismatch", "review", f"form {_fmt(f_area)}/{_fmt(f_loss)} vs app {_fmt(app_area)}/{_fmt(app_loss)}")
                parts_form.append(s + ".")
            if source == "total row" and ra_ok and rl_ok and ra is not None and rl is not None and (abs(ra - fa) > 0.5 or abs(rl - fl) > 0.5):
                parts_form.append(f"Table row shows {_fmt(ra)}% / {_fmt(rl)}% but the total row shows {_fmt(fa)}% / {_fmt(fl)}% - inconsistent.")
                flag("Form row/total inconsistent", "review")
            if total_blank and source != "table row":
                parts_form.append("Total row blank on form.")
            # signatures
            if form.get("template") == "rajasthan":
                sig = [("farmer", form.get("farmer_signed")), ("insurance company", form.get("company_signed")),
                       ("agriculture supervisor (AAO)", form.get("officer_signed"))]
            else:
                sig = [("farmer", form.get("farmer_signed")), ("company", form.get("company_signed")),
                       ("primary worker", form.get("worker_signed")), ("block officer", form.get("officer_signed"))]
            bits = []
            missing = []
            for name, v in sig:
                if v is None:
                    bits.append(f"{name} not assessed" if name in ("block officer", "agriculture supervisor (AAO)") else f"{name} not readable")
                else:
                    bits.append(f"{name} {'Yes' if v else 'NO'}")
                    if not v:
                        missing.append(name)
            parts_form.append("Signatures: " + ", ".join(bits) + ".")
            # ---- the four handwritten dates (only when the date reader produced at least one of them)
            fd = {k: _parse_date(form.get(k)) for k in ("sow_date", "loss_date", "intimation_date", "inspection_date")}
            if any(fd.values()):
                nm = {"sow_date": "sowing", "loss_date": "loss", "intimation_date": "intimation", "inspection_date": "inspection"}
                got = [f"{nm[k]} {v:%d-%m-%Y}" for k, v in fd.items() if v]
                n_missing = sum(1 for v in fd.values() if not v)
                parts_form.append("Dates on form: " + ", ".join(got) + (f" ({n_missing} other date(s) not readable)" if n_missing else "") + ".")
                a_loss, a_int = _parse_date(row.get("survey_start_date")), _parse_date(row.get("survey_end_date"))
                if fd["loss_date"] and a_loss and fd["loss_date"] != a_loss:
                    parts_form.append(f"Loss date on form {fd['loss_date']:%d-%m-%Y} differs from the app ({a_loss:%d-%m-%Y}).")
                    flag("Form date differs from app", "partial")   # a mismatch like the PO ID: Partially OK
                if fd["intimation_date"] and a_int and fd["intimation_date"] != a_int:
                    parts_form.append(f"Intimation date on form {fd['intimation_date']:%d-%m-%Y} differs from the app ({a_int:%d-%m-%Y}).")
                    flag("Form date differs from app", "partial")
                seq = [fd[k] for k in ("sow_date", "loss_date", "intimation_date", "inspection_date") if fd[k]]
                if any(b < a for a, b in zip(seq, seq[1:])):
                    parts_form.append("Dates on the form are not in the usual order (sowing, loss, intimation, inspection).")
                    flag("Form dates out of order", "partial")
            # The primary worker and the block officer almost never sign (officer: 0 of 149 labelled forms; worker: ~17%), so their
            # absence is informational only. A missing FARMER or COMPANY signature is what needs a review.
            core_missing = [m for m in missing if m in ("farmer", "company", "insurance company", "agriculture supervisor (AAO)")]
            if core_missing:
                flag("Signature missing", None, "signature missing: " + ", ".join(core_missing))   # user's rule: noted in the remark, verdict stays OK
            if form.get("officer_stamp_only"):
                parts_form.append("Block officer block has a rubber stamp only (not a signature).")
                flag("Officer stamp only", "review")
            if form.get("overwrite_suspected"):
                parts_form.append("Overwriting / cutting suspected on the form.")
                flag("Form overwrite", "review", "overwriting suspected on the form")
            if conf_low:
                flag("Low confidence - manual review", "manual", "form reading confidence low")
                parts_form.append(f"Form reading confidence low ({conf:.2f}).")
            for n in (form.get("notes") or [])[:2]:
                parts_form.append(str(n).rstrip(".") + ".")

    # ------------------------------------------------------------ PHOTOS
    pstate = (photos or {}).get("_state") if photos else None
    if pstate is None:
        pstate = "no_link" if (photos is None and photo_url_blank) else ("not_found" if photos is None or photos.get("_error") or photos.get("photo_status") in ("Unavailable", "Error") else "ok")
    field_photo = None
    photo_dist = None
    if pstate == "skipped":
        pass
    elif pstate == "no_link":
        flag("Photo link missing", "reject", "no photo link in the record")
        parts_photo.append("No photo link in the record.")
    elif pstate in ("not_found", "error"):
        err = (photos or {}).get("_error") or (photos or {}).get("photo_error")
        flag("Photos not found", "manual", "photos could not be retrieved")
        parts_photo.append("Photos could not be retrieved" + (f" ({err})" if err else "") + ".")
    else:
        n = int(_num(photos.get("_n_photos")) or _num(photos.get("n_photos")) or 0)
        n_form = int(_num(photos.get("n_form_photos")) or 0)
        scene = photos.get("scene_type")
        is_form = bool(photos.get("photo_is_form")) or scene == "paper form"
        field_photo = photos.get("field_photo")
        if is_form:
            d = _parse_date(photos.get("stamp_date") or photos.get("photo_date"))
            ref = _parse_date(form.get("inspection_date")) if form and form.get("_state") in (None, "ok") else None
            ref_txt = "inspection"
            if ref is None:
                ref, ref_txt = _parse_date(row.get("survey_start_date")), "loss date"
            cnt = f"all {n}" if n and n_form >= n else (f"{n_form} of {n}" if n else "the")
            s = f"{cnt} uploaded photos are pictures of the paper form - no field photograph uploaded"
            if d:
                s += f" (photo date {d:%d-%m-%Y}"
                if ref:
                    days = (d - ref).days
                    s += f", {abs(days)} days {'after' if days >= 0 else 'before'} {ref_txt}"
                s += ")"
            parts_photo.append(s + ".")
            # only a hard reject when no real field photo accompanies it
            if not n or n_form >= n:
                flag("Photo is form image", "reject", "photographs are pictures of the form, not the field")
            else:
                flag("Photo is form image", "review", f"{n_form} photo(s) are pictures of the form")
        else:
            label = {"no crop": "no crop visible", "cut & spread": "cut & spread crop", "crop mismatch": "crop does not match",
                     "standing crop": "standing crop"}.get(field_photo, field_photo)
            s = f"{n or 'Photos'} photo(s) analysed" if n else "Photos analysed"
            if scene:
                s += f"; scene: {scene}"
            cs, cconf = photos.get("crop_seen"), _num(photos.get("crop_seen_conf"))
            declared = row.get("crop_name")
            # Photo-content claims (crop type / flooding / damage / no crop) are only ~60-95% accurate, so they are
            # asserted as facts only when both independent photo models agree and confidence is high.
            trusted = photos.get("photo_conf") == "high" and photos.get("photo_agree") is not False
            if cs:
                lowc = (cconf is not None and cconf < LOW_FIELD_CONF) or not trusted
                s += f"; crop seen: {cs}" + (" (low confidence)" if lowc else "")
                if photos.get("crop_matches_declared") is False and not lowc:
                    s += f" - does NOT match declared crop{'' if _blank(declared) else ' ' + str(declared)}"
                    flag("Crop mismatch", "review", "crop in the photo differs from the declared crop")
                elif photos.get("crop_matches_declared") is True:
                    s += " (matches declared crop)"
            if photos.get("flooded") in (True, "yes"):
                wf = _num(photos.get("water_frac"))
                s += ("; flooding/waterlogging seen" if trusted else "; possible flooding (low confidence - manual check)") + (f" (~{wf * 100:.0f}% of the frame)" if wf else "")
                flag("Flooding seen", None)
            if photos.get("damage_state"):
                s += (f"; damage state: {photos['damage_state']}" if trusted
                      else f"; possible damage state: {photos['damage_state']} (low confidence - manual check)")
            if label:
                s += f"; field state: {label} (heuristic)"
            parts_photo.append(s + ".")
            cp = photos.get("crop_present")
            crop_absent = cp in (False, "no") or (cp is None and field_photo == "no crop")
            if crop_absent and scene in (None, "field") and not trusted:
                parts_photo.append("Crop may be absent in the photo (low confidence - manual check).")
            elif crop_absent and scene in (None, "field"):
                parts_photo.append("No crop visible in the photo.")
                low_claim = app_loss is not None and app_loss <= 50
                flag("No crop in photo", "review" if low_claim else None,
                     f"no crop visible but reported loss is only {_fmt(app_loss)}%" if low_claim else None)
            if scene in ("person-only", "house-road-sky-other"):
                parts_photo.append(f"Photo does not show the field (scene: {scene}).")
                flag("Photo not of the field", "review", "photo does not show the field")
            elif scene == "blurry-dark-irrelevant":
                parts_photo.append("Photo too blurry/dark/irrelevant to judge.")
                flag("Photo not of the field", "manual", "photos unusable (blurry/dark/irrelevant)")
        # is a person (the farmer) visible in any field photo of this docket?
        each = photos.get("person_each")
        if isinstance(each, (list, tuple)) and each and not is_form:
            isf_ = photos.get("photo_is_form_each") or []
            field_ = [i for i in range(len(each)) if not (i < len(isf_) and isf_[i])]
            seen_ = [i + 1 for i in field_ if each[i]]
            if seen_:
                parts_photo.append(f"Farmer available: person visible in photo {', '.join(map(str, seen_))} of {len(each)}.")
            elif field_:
                parts_photo.append(f"Farmer not available: no person with at least 25% of the body visible in the {len(field_)} field photo(s).")
        # duplicates / rotation
        nd = int(_num(photos.get("n_duplicates")) or 0)
        if nd:
            parts_photo.append(f"{nd} duplicate photo(s) in this record.")
            flag("Duplicate photos", None)   # usually the same scene re-shot, not fraud: informational
        if photos.get("rotated"):
            parts_photo.append("Photo(s) are rotated 90 degrees.")
            flag("Photos rotated", None)   # auto-rotated for analysis: informational
        # GPS stamp vs app
        dist = _num(photos.get("stamp_dist_m")) if C.USE_PHOTO_GPS else None
        photo_dist = dist
        if dist is not None:
            if dist > C.GPS_PHOTO_MAX_DISTANCE_M:
                parts_photo.append(f"GPS stamp {dist:,.0f} m from the app location (limit {C.GPS_PHOTO_MAX_DISTANCE_M} m).")
                flag("Photo GPS mismatch", "review", f"photo GPS {dist:,.0f} m from app point")
            else:
                parts_photo.append(f"GPS stamp {dist:,.0f} m from app.")
        elif not is_form and C.USE_PHOTO_GPS:
            parts_photo.append("No readable GPS stamp on the photos.")
        # date
        pd_ = _parse_date(photos.get("stamp_date") or photos.get("photo_date"))
        if pd_ and not is_form:
            ls, le = _parse_date(row.get("survey_start_date")), _parse_date(row.get("survey_end_date"))
            ds = [x for x in (ls, le) if x]
            if ds and (pd_ < min(ds) or (pd_ - max(ds)).days > 30):
                parts_photo.append(f"Photo date {pd_:%d-%m-%Y} is outside the survey window.")
                flag("Photo date outside survey period", "review")
            else:
                parts_photo.append(f"Photo date {pd_:%d-%m-%Y}.")
        # field state vs reported loss
        est = _num(photos.get("photo_loss"))
        dstate = photos.get("damage_state")
        if not is_form and app_loss is not None and dstate and not (photos.get("photo_conf") == "high" and photos.get("photo_agree") is not False):
            pass  # uncertain photo reading: no conflict flag, manual check is already suggested in the photo remark
        elif not is_form and app_loss is not None and dstate:
            if dstate == "healthy" and app_loss >= 50:
                parts_photo.append(f"Photo shows a healthy crop but reported crop loss is {_fmt(app_loss)}%.")
                flag("Field state vs reported loss", "review")
            elif dstate in ("lodged", "submerged", "dried-burnt", "cut & spread", "partly damaged") and app_loss <= 10 and (app_area or 0) <= 10:
                parts_photo.append(f"Photo shows damage ({dstate}) but reported crop loss is only {_fmt(app_loss)}%.")
                flag("Field state vs reported loss", "review")
        if not is_form and app_loss is not None and not dstate:
            if field_photo == "standing crop" and app_loss >= 50 and (est is None or est < app_loss - C.PHOTO_LOSS_DIFF_FLAG_PCT):
                parts_photo.append(f"Field looks like standing crop but reported crop loss is {_fmt(app_loss)}%.")
                flag("Field state vs reported loss", "review")
            elif field_photo in ("no crop", "cut & spread") and app_loss <= 10:
                parts_photo.append(f"Field shows {field_photo} but reported crop loss is only {_fmt(app_loss)}%.")
                flag("Field state vs reported loss", "review")
        if photos.get("photo_quality") in ("blurry", "dark", "irrelevant"):
            parts_photo.append(f"Photo quality: {photos['photo_quality']}.")
        for t in (photos.get("remarks") or [])[:2]:
            t = str(t).strip().rstrip(".")
            if t and t not in " ".join(parts_photo):
                parts_photo.append(t + ".")

    # ------------------------------------------------------------ GPS cluster
    same_txt = gps.get("Same_Location_Remark")
    same_txt = None if _blank(same_txt) else str(same_txt).strip()
    g_rem = str(gps.get("Suggested_Remark") or "OK")
    same = int(_num(gps.get("Nearby_Same_Surveyor_25m")) or 0)
    onf = int(_num(gps.get("Records_On_Same_Field")) or 0)
    dmg = _num(row.get("total_damage_pct"))
    if same_txt:
        flag("Same location", "review")   # user's rule (2026-10-03): several surveys at the same spot -> Review
    sv_txt = str(gps.get("Same_Location_Values") or "")
    if "possible copied entry" in sv_txt:
        flag("Same app entry at same spot", "review")
        parts_gps.append("Surveys at the same spot carry the same app entry - possible copied entry.")
    if g_rem.startswith("Same Location"):
        if same_txt:
            pass
        elif "QC Required" in g_rem:
            parts_gps.append(f"{same} same-surveyor records within {C.GPS_PROXIMITY_RADIUS_M} m.")
        else:
            parts_gps.append(f"{same} same-surveyor records within {C.GPS_PROXIMITY_RADIUS_M} m (low damage{'' if dmg is None else f' {_fmt(dmg)}%'}).")
        flag("GPS cluster", "review")
    elif g_rem.startswith("Review"):
        parts_gps.append(f"{onf} records on the same survey number.")
        flag("GPS cluster", "review")

    # ------------------------------------------------------------ data flags
    dflags = [x.strip() for x in str(data_flags or "").split(",") if x.strip() and x.strip() not in COVERED_DATA_FLAGS]
    if dflags:
        flags.extend("Data: " + x for x in dflags)
        place = any(f in flags for f in ("Same location", "GPS cluster", "Same app entry at same spot"))
        # 'Possible duplicate field' (same khasra in another docket) says the same as the same-location finding when that is
        # present: then it is informational like the location flags; on its own (or with other data flags) it needs review
        if [x for x in dflags if not (place and x == "Possible duplicate field")]:
            severity["review"].append("data")
    # ------------------------------------------------------------ risk
    rscore = _num(risk.get("Risk_Score"))
    rtxt = ""
    if rscore is not None and rscore >= RISK_REVIEW:
        reasons = str(risk.get("Risk_Reasons") or "").strip()
        rtxt = f"RISK {rscore:.0f}" + (f": {reasons}." if reasons else ".")
        flags.append(f"High risk score ({rscore:.0f})")
        severity["review"].append("risk")

    # ------------------------------------------------------------ verdict + confidence
    conf_form = _num((form or {}).get("confidence")) if form and fstate == "ok" else None
    if severity["reject"] or severity["manual"]:
        verdict = "Manual QC Required"
    elif severity["review"]:
        verdict = "Review"
    elif severity["partial"]:
        verdict = "Partially OK"
    else:
        verdict = "OK"          # informational findings (signature missing, duplicate photos, ...) only go into the remark
    evidence_failed = fstate in ("no_link", "not_found", "error") or pstate in ("no_link", "not_found", "error")
    if evidence_failed or severity["manual"] or severity["reject"] or (conf_form is not None and conf_form < C.LOW_CONFIDENCE_THRESHOLD) or \
            (form and fstate == "ok" and form.get("is_proforma3") is False):
        confidence = "Low"
    elif severity["review"] or (conf_form is not None and conf_form < 0.8) or fstate == "skipped" or pstate == "skipped":
        confidence = "Medium"
    else:
        confidence = "High"

    sections = []
    if parts_form:
        sections.append("FORM: " + " ".join(parts_form))
    if parts_photo:
        sections.append("PHOTOS: " + " ".join(parts_photo))
    if same_txt:
        sections.append("SAME LOCATION: " + same_txt)
    if parts_gps:
        sections.append("GPS: " + " ".join(parts_gps))
    if dflags:
        sections.append("DATA: " + "; ".join(dflags) + ".")
    if rtxt:
        sections.append(rtxt)
    head = ""
    if verdict not in ("OK", "Partially OK") and headline:
        head = f"{verdict}: " + "; ".join(dict.fromkeys(headline[:3])) + ". "
    elif verdict == "Partially OK":
        head = "Partially OK. "
    elif verdict == "OK":
        head = "OK: no issues found. "
    detail = head + (" | ".join(sections))
    flags = list(dict.fromkeys(flags))
    counters = sorted({COUNTER_OF[f] for f in flags if f in COUNTER_OF})
    try:
        remark = _plain(row, form, photos, gps, flags, verdict, {"form_area": f_area, "form_loss": f_loss, "match": match})
    except Exception:       # the plain text is cosmetic; fall back to the detailed one
        remark = detail
    return {"remark": remark.strip(), "remark_detail": detail.strip(), "verdict": verdict, "confidence": confidence, "flags": flags, "counters": counters,
            "match": match, "form_area": f_area, "form_loss": f_loss, "form_source": source,
            "field_photo": field_photo, "photo_dist_m": photo_dist}


def build_remark(row, form=None, photos=None, gps=None, data_flags="", risk=None) -> str:
    """Prioritised, factual, multi-sentence remark for one row (see module docstring for the verdict rules)."""
    return evaluate(row, form, photos, gps, data_flags, risk)["remark"]
