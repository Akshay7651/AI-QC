"""Row-level QC reasoning: turns what the form reader / photo analyst / GPS / data / risk engines found into
  * a precise, prioritised, human-readable remark  ('Any Other Remarks')
  * a QC Verdict  (OK / Review / Reject-evidence / Manual-check)
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

VERDICTS = ("OK", "Review", "Reject-evidence", "Manual-check")
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
def evaluate(row, form=None, photos=None, gps=None, data_flags="", risk=None):
    """Return dict(remark, verdict, confidence, flags, counters, match, form_area, form_loss, form_source)."""
    row = row or {}
    gps = gps or {}
    risk = risk or {}
    flags: list[str] = []          # short tags (AI_Flags)
    severity = {"reject": [], "manual": [], "review": []}
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
            s = f"Form No {fn}" if fn and fn_ok else "Form No not readable"
            po = form.get("po_id_matches")
            if po is True:
                s += "; PO ID matches docket"
            elif po is False:
                flag("PO ID mismatch", "review", "PO ID on the form differs from the docket")
                shown = form.get("po_id") if _readable(form, form.get("po_id"), "po_id") else None
                s += f"; PO ID {shown} does NOT match docket" if shown else "; PO ID does NOT match docket"
            else:
                s += "; PO ID not readable"
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
            if f_area is None:
                parts_form.append("Affected area % / loss % not readable on the form.")
                if not conf_low:
                    flag("Form not readable", "manual")
            else:
                if source == "total row":
                    s = f"Form total affected area {_fmt(f_area)}% / loss {_fmt(f_loss)}%"
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
            sig = [("farmer", form.get("farmer_signed")), ("company", form.get("company_signed")),
                   ("primary worker", form.get("worker_signed")), ("block officer", form.get("officer_signed"))]
            bits = []
            missing = []
            for name, v in sig:
                if v is None:
                    bits.append(f"{name} not readable")
                else:
                    bits.append(f"{name} {'Yes' if v else 'NO'}")
                    if not v:
                        missing.append(name)
            parts_form.append("Signatures: " + ", ".join(bits) + ".")
            if missing:
                flag("Signature missing", "review", "signature missing: " + ", ".join(missing))
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
            if cs:
                lowc = cconf is not None and cconf < LOW_FIELD_CONF
                s += f"; crop seen: {cs}" + (" (low confidence)" if lowc else "")
                if photos.get("crop_matches_declared") is False and not lowc:
                    s += f" - does NOT match declared crop{'' if _blank(declared) else ' ' + str(declared)}"
                    flag("Crop mismatch", "review", "crop in the photo differs from the declared crop")
                elif photos.get("crop_matches_declared") is True:
                    s += " (matches declared crop)"
            if photos.get("flooded"):
                wf = _num(photos.get("water_frac"))
                s += "; flooding/waterlogging seen" + (f" (~{wf * 100:.0f}% of the frame)" if wf else "")
                flag("Flooding seen", None)
            if photos.get("damage_state"):
                s += f"; damage state: {photos['damage_state']}"
            if label:
                s += f"; field state: {label} (heuristic)"
            parts_photo.append(s + ".")
            crop_absent = photos.get("crop_present") is False or (photos.get("crop_present") is None and field_photo == "no crop")
            if crop_absent and scene in (None, "field"):
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
        # duplicates / rotation
        nd = int(_num(photos.get("n_duplicates")) or 0)
        if nd:
            parts_photo.append(f"{nd} duplicate photo(s) in this record.")
            flag("Duplicate photos", "review")
        if photos.get("rotated"):
            parts_photo.append("Photo(s) are rotated 90 degrees.")
            flag("Photos rotated", "review")
        # GPS stamp vs app
        dist = _num(photos.get("stamp_dist_m"))
        photo_dist = dist
        if dist is not None:
            if dist > C.GPS_PHOTO_MAX_DISTANCE_M:
                parts_photo.append(f"GPS stamp {dist:,.0f} m from the app location (limit {C.GPS_PHOTO_MAX_DISTANCE_M} m).")
                flag("Photo GPS mismatch", "review", f"photo GPS {dist:,.0f} m from app point")
            else:
                parts_photo.append(f"GPS stamp {dist:,.0f} m from app.")
        elif not is_form:
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
        if not is_form and app_loss is not None and dstate:
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
    g_rem = str(gps.get("Suggested_Remark") or "OK")
    same = int(_num(gps.get("Nearby_Same_Surveyor_25m")) or 0)
    onf = int(_num(gps.get("Records_On_Same_Field")) or 0)
    dmg = _num(row.get("total_damage_pct"))
    if g_rem.startswith("Same Location"):
        if "QC Required" in g_rem:
            parts_gps.append(f"{same} same-surveyor records within {C.GPS_PROXIMITY_RADIUS_M} m - QC required.")
        else:
            parts_gps.append(f"{same} same-surveyor records within {C.GPS_PROXIMITY_RADIUS_M} m (low damage{'' if dmg is None else f' {_fmt(dmg)}%'}).")
        flag("GPS cluster", "review", f"{same} same-surveyor records within {C.GPS_PROXIMITY_RADIUS_M} m")
    elif g_rem.startswith("Review"):
        parts_gps.append(f"{onf} records on the same survey number - review.")
        flag("GPS cluster", "review")

    # ------------------------------------------------------------ data flags
    dflags = [x.strip() for x in str(data_flags or "").split(",") if x.strip() and x.strip() not in COVERED_DATA_FLAGS]
    if dflags:
        flags.extend("Data: " + x for x in dflags)
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
    if severity["reject"]:
        verdict = "Reject-evidence"
    elif severity["manual"]:
        verdict = "Manual-check"
    elif severity["review"]:
        verdict = "Review"
    else:
        verdict = "OK"
    evidence_failed = fstate in ("no_link", "not_found", "error") or pstate in ("no_link", "not_found", "error")
    if evidence_failed or severity["manual"] or (conf_form is not None and conf_form < C.LOW_CONFIDENCE_THRESHOLD) or \
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
    if parts_gps:
        sections.append("GPS: " + " ".join(parts_gps))
    if dflags:
        sections.append("DATA: " + "; ".join(dflags) + ".")
    if rtxt:
        sections.append(rtxt)
    head = ""
    if verdict != "OK" and headline:
        head = f"{verdict.upper()}: " + "; ".join(dict.fromkeys(headline[:3])) + ". "
    elif verdict == "OK":
        head = "OK: no issues found. "
    remark = head + (" | ".join(sections))
    flags = list(dict.fromkeys(flags))
    counters = sorted({COUNTER_OF[f] for f in flags if f in COUNTER_OF})
    return {"remark": remark.strip(), "verdict": verdict, "confidence": confidence, "flags": flags, "counters": counters,
            "match": match, "form_area": f_area, "form_loss": f_loss, "form_source": source,
            "field_photo": field_photo, "photo_dist_m": photo_dist}


def build_remark(row, form=None, photos=None, gps=None, data_flags="", risk=None) -> str:
    """Prioritised, factual, multi-sentence remark for one row (see module docstring for the verdict rules)."""
    return evaluate(row, form, photos, gps, data_flags, risk)["remark"]
