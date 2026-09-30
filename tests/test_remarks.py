import remarks as R


def row(**kw):
    r = {"docket_id": "040106260000636479", "affected_area_pct": 0, "crop_loss_pct": 0, "total_damage_pct": 0,
         "survey_start_date": "2026-09-10", "survey_end_date": "2026-09-12", "pdf_url": "http://x/f.pdf", "media_urls": "http://x/p.jpg"}
    r.update(kw)
    return r


def form(**kw):
    f = {"_state": "ok", "is_proforma3": True, "quality": "good", "form_no": "HR0126175631", "po_id": "040106260000636479",
         "po_id_matches": True, "form_area": 0.0, "form_loss": 0.0, "row_area": 0.0, "row_loss": 0.0, "total_row_blank": False,
         "farmer_signed": True, "company_signed": True, "worker_signed": True, "officer_signed": True, "officer_stamp_only": False,
         "overwrite_suspected": False, "confidence": 0.9, "field_conf": {}, "notes": [], "inspection_date": "15092026"}
    f.update(kw)
    return f


def photos(**kw):
    p = {"_state": "ok", "_n_photos": 3, "photo_status": "OK", "field_photo": "standing crop", "photo_is_form": False,
         "n_form_photos": 0, "n_duplicates": 0, "stamp_dist_m": 6.0, "stamp_date": "14092026", "rotated": False, "remarks": [],
         "photo_loss": 0, "photo_conf": "high", "photo_agree": True}
    p.update(kw)
    return p


def ev(r=None, f=None, p=None, **kw):
    return R.evaluate(r or row(), f if f is not None else form(), p if p is not None else photos(), **kw)


def test_clean_row_is_ok_high():
    e = ev()
    assert e["verdict"] == "OK" and e["confidence"] == "High" and e["flags"] == []
    assert "Form No HR0126175631; PO ID matches docket" in e["remark"]
    assert "= app (Match)" in e["remark"] and e["match"] == "Match"
    assert "Signatures: farmer Yes, company Yes, primary worker Yes, block officer Yes." in e["remark"]


def test_the_brief_example_shape():
    f = form(worker_signed=False, officer_signed=False, total_row_blank=True, form_area=0.0, form_loss=0.0)
    p = photos(photo_is_form=True, n_form_photos=5, _n_photos=5, stamp_date="27092026", stamp_dist_m=6.0)
    gps = {"Suggested_Remark": "Same Location - QC Required", "Nearby_Same_Surveyor_25m": 308}
    risk = {"Risk_Score": 72, "Risk_Reasons": "surveyor averages 185 records/day"}
    e = R.evaluate(row(), f, p, gps, "", risk)
    t = e["remark"]
    assert e["verdict"] == "Reject-evidence"
    assert "primary worker NO, block officer NO" in t and "Total row blank on form." in t
    assert "all 5 uploaded photos are pictures of the paper form - no field photograph uploaded" in t
    assert "photo date 27-09-2026, 12 days after inspection" in t
    assert "GPS stamp 6 m from app" in t
    assert "308 same-surveyor records within 25 m - QC required" in t
    assert "RISK 72: surveyor averages 185 records/day." in t
    assert t.index("FORM:") < t.index("PHOTOS:") < t.index("GPS:") < t.index("RISK 72")
    assert {"photo_is_form", "signature_missing", "gps_cluster"} <= set(e["counters"])


def test_missing_links_reject():
    e = R.evaluate(row(pdf_url=None), None, photos())
    assert e["verdict"] == "Reject-evidence" and "missing_form_link" in e["counters"] and "No signed-form link" in e["remark"]
    e = R.evaluate(row(media_urls=""), form(), None)
    assert e["verdict"] == "Reject-evidence" and "missing_photo_link" in e["counters"]


def test_download_failed_is_manual_not_reject():
    e = R.evaluate(row(), {"_state": "not_found", "_error": "HTTP 404"}, photos())
    assert e["verdict"] == "Manual-check" and e["confidence"] == "Low" and "HTTP 404" in e["remark"] and "form_not_found" in e["counters"]


def test_not_proforma3_rejects():
    e = ev(f=form(is_proforma3=False, quality="blurry"))
    assert e["verdict"] == "Reject-evidence" and "not a Proforma-3" in e["remark"] and "not_proforma3" in e["counters"]


def test_mismatch_review_and_values_quoted():
    e = ev(r=row(affected_area_pct=30, crop_loss_pct=50), f=form(form_area=0.0, form_loss=0.0))
    assert e["verdict"] == "Review" and e["match"] == "Mismatch"
    assert "vs app 30% / 50% (Mismatch)" in e["remark"] and "mismatch" in e["counters"]


def test_within_tolerance_matches():
    e = ev(r=row(affected_area_pct=30, crop_loss_pct=50), f=form(form_area=32.0, form_loss=48.0))
    assert e["match"] == "Match"


def test_low_confidence_value_is_not_read_and_never_invented():
    f = form(form_area=70.0, form_loss=70.0, row_area=None, row_loss=None, field_conf={"area": 0.2, "loss": 0.2})
    e = ev(f=f)
    assert "not readable" in e["remark"] and "70" not in e["remark"].split("FORM:")[1].split("Signatures")[0]
    assert e["match"] == "NA" and e["verdict"] == "Manual-check"


def test_row_vs_total_inconsistent_and_blank_total_uses_row():
    e = ev(f=form(form_area=30.0, form_loss=30.0, row_area=0.0, row_loss=0.0))
    assert "inconsistent" in e["remark"] and e["verdict"] == "Review"
    e = ev(f=form(form_area=None, form_loss=None, total_row_blank=True, row_area=0.0, row_loss=0.0))
    assert "table row shows affected area 0% / loss 0%" in e["remark"] and e["match"] == "Match"


def test_signature_missing_and_stamp_only_and_overwrite_and_po():
    e = ev(f=form(farmer_signed=False, officer_signed=False, officer_stamp_only=True, overwrite_suspected=True,
                  po_id_matches=False, po_id="040106260000000001"))
    t = e["remark"]
    assert "farmer NO" in t and "rubber stamp only" in t and "Overwriting" in t and "does NOT match docket" in t
    assert e["verdict"] == "Review" and {"signature_missing", "overwrite", "po_id_mismatch"} <= set(e["counters"])


def test_unknown_signature_is_not_readable_not_no():
    e = ev(f=form(worker_signed=None))
    assert "primary worker not readable" in e["remark"] and e["verdict"] == "OK"


def test_photo_gps_date_duplicates_rotated():
    e = ev(p=photos(stamp_dist_m=1500.0, n_duplicates=2, rotated=True, stamp_date="01082026"))
    t = e["remark"]
    assert "1,500 m from the app location" in t and "2 duplicate" in t and "rotated" in t and "outside the survey window" in t
    assert e["verdict"] == "Review" and {"gps_mismatch", "duplicate_photos"} <= set(e["counters"])


def test_some_photos_form_is_review_not_reject():
    e = ev(p=photos(photo_is_form=True, n_form_photos=2, _n_photos=5))
    assert e["verdict"] == "Review" and "2 of 5" in e["remark"]


def test_field_state_vs_reported_loss():
    e = ev(r=row(crop_loss_pct=80, affected_area_pct=80), f=form(form_area=80.0, form_loss=80.0), p=photos(photo_loss=0))
    assert "standing crop but reported crop loss is 80%" in e["remark"] and e["verdict"] == "Review"


def test_data_flags_gps_cluster_and_risk_thresholds():
    e = ev(data_flags="Missing: farmer_name, Missing: signed form link, Duplicate record")
    assert "DATA: Missing: farmer_name; Duplicate record." in e["remark"] and "signed form link" not in e["remark"].split("DATA:")[1]
    e = ev(gps={"Suggested_Remark": "Review - Multiple Records", "Records_On_Same_Field": 6})
    assert "6 records on the same survey number" in e["remark"]
    assert ev(risk={"Risk_Score": 59, "Risk_Reasons": "x"})["verdict"] == "OK"
    e = ev(risk={"Risk_Score": 61, "Risk_Reasons": "x"})
    assert e["verdict"] == "Review" and "RISK 61: x." in e["remark"]


def test_low_overall_confidence_is_manual_check():
    e = ev(f=form(confidence=0.4))
    assert e["verdict"] == "Manual-check" and e["confidence"] == "Low" and "manual" in " ".join(e["flags"]).lower()
    assert ev(f=form(confidence=0.7))["confidence"] == "Medium"


def test_priority_reject_beats_manual_beats_review():
    e = ev(f=form(confidence=0.3, farmer_signed=False), p=photos(photo_is_form=True, n_form_photos=3, _n_photos=3))
    assert e["verdict"] == "Reject-evidence"
    e = ev(f=form(confidence=0.3, farmer_signed=False))
    assert e["verdict"] == "Manual-check"


def test_skipped_parts_are_not_reported():
    e = R.evaluate(row(), {"_state": "skipped"}, photos())
    assert "FORM:" not in e["remark"] and e["confidence"] == "Medium"


def test_build_remark_matches_evaluate_and_handles_garbage():
    assert R.build_remark(row(), form(), photos()) == ev()["remark"]
    assert isinstance(R.build_remark({}, None, None), str)
    e = R.evaluate({"affected_area_pct": float("nan")}, form(), photos(stamp_dist_m=float("nan")), {"Nearby_Same_Surveyor_25m": float("nan")}, None, {"Risk_Score": float("nan")})
    assert e["verdict"] in R.VERDICTS


def test_crop_scene_flood_damage_fields():
    p = photos(scene_type="field", crop_present=True, crop_seen="maize", crop_seen_conf=0.8, crop_matches_declared=False,
               flooded=True, water_frac=0.3, damage_state="submerged")
    e = ev(r=row(crop_name="Paddy", crop_loss_pct=60, affected_area_pct=60), f=form(form_area=60.0, form_loss=60.0), p=p)
    t = e["remark"]
    assert "crop seen: maize - does NOT match declared crop Paddy" in t and "flooding/waterlogging seen (~30% of the frame)" in t
    assert "damage state: submerged" in t and "scene: field" in t
    assert e["verdict"] == "Review" and {"crop_mismatch", "flooded"} <= set(e["counters"])
    # low-confidence crop id never produces a mismatch flag
    e = ev(p=photos(crop_seen="maize", crop_seen_conf=0.3, crop_matches_declared=False))
    assert "crop_mismatch" not in e["counters"] and "(low confidence)" in e["remark"] and e["verdict"] == "OK"


def test_no_crop_only_flags_when_reported_loss_is_low():
    e = ev(p=photos(crop_present=False, field_photo="no crop"))                       # loss 0 -> inconsistent
    assert e["verdict"] == "Review" and "no_crop_in_photo" in e["counters"]
    e = ev(r=row(crop_loss_pct=90, affected_area_pct=90), f=form(form_area=90.0, form_loss=90.0, row_area=90.0, row_loss=90.0), p=photos(crop_present=False, field_photo="no crop"))
    assert e["verdict"] == "OK" and "no_crop_in_photo" in e["counters"]              # consistent with a total loss: counted, not a flag


def test_scene_types():
    assert ev(p=photos(scene_type="paper form", photo_is_form=False, n_form_photos=3, _n_photos=3))["verdict"] == "Reject-evidence"
    assert ev(p=photos(scene_type="house-road-sky-other"))["verdict"] == "Review"
    e = ev(p=photos(scene_type="blurry-dark-irrelevant"))
    assert e["verdict"] == "Manual-check" and "photo_not_field" in e["counters"]


def test_damage_state_vs_reported_loss():
    e = ev(r=row(crop_loss_pct=80, affected_area_pct=80), f=form(form_area=80.0, form_loss=80.0, row_area=80.0, row_loss=80.0), p=photos(damage_state="healthy"))
    assert e["verdict"] == "Review" and "healthy crop but reported crop loss is 80%" in e["remark"]
    assert ev(p=photos(damage_state="lodged"))["verdict"] == "Review"      # app loss 0 vs visible damage


def test_same_location_remark_verbatim():
    txt = "MULTIPLE SURVEYS AT SAME LOCATION: 448 other record(s) within 25 m (448 by the same surveyor) - cluster G-251"
    e = ev(gps={"Suggested_Remark": "Same Location - QC Required", "Nearby_Same_Surveyor_25m": 448, "Same_Location_Remark": txt})
    assert "SAME LOCATION: " + txt in e["remark"] and e["verdict"] == "Review" and "same_location" in e["counters"]
    assert "448 same-surveyor records within" not in e["remark"]


def test_uncertain_photo_claims_are_not_asserted():
    """Crop type / flooding / damage / no-crop are only asserted when both photo models agree with high confidence."""
    p = photos(scene_type="field", crop_present=False, crop_seen="maize", crop_seen_conf=0.9, crop_matches_declared=False,
               flooded=True, damage_state="lodged", photo_conf="low", photo_agree=False)
    e = ev(p=p)
    assert "low confidence - manual check" in e["remark"]
    assert "does NOT match declared crop" not in e["remark"]
    assert "Crop mismatch" not in e["flags"] and "No crop in photo" not in e["flags"]
    assert "Field state vs reported loss" not in e["flags"]
