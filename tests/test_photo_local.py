"""Tests for photo_local / photo_stamp / photo_feats (synthetic images only; no data/ needed)."""
import os

import cv2
import numpy as np
import pytest

import photo_feats as PF
import photo_local as PL
import photo_stamp as PS


def _stamp_img(lat="29.4106963", lng="75.8701606", date="14-08-2026 05:31 PM", bg=90):
    img = np.full((1280, 720, 3), bg, np.uint8)
    rng = np.random.default_rng(0)
    img[:1000] = rng.integers(40, 160, (1000, 720, 3), dtype=np.uint8)
    y = 1110
    for t in (f"Latitude: {lat}", f"Longitude: {lng}", "GPS Accuracy: 20.0 (mtr)", f"Date: {date}"):
        cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += 38
    return img


def test_parse_block_handles_missing_spaces_and_dot():
    p = PS.parse_block("Latitude:29.4106963\nLongitude:75.8701606\nGPSAccuracy:20.0(mtr)\nDate:14-08-202605:31PM\n")
    assert abs(p["lat"] - 29.4106963) < 1e-7 and abs(p["lng"] - 75.8701606) < 1e-7
    assert p["date"] == "14082026" and p["time"] == "17:31"
    p = PS.parse_block("Latitude:281246425\nLongitude:77.3275832\n")      # lost decimal point
    assert abs(p["lat"] - 28.1246425) < 1e-7
    assert PS.parse_block("Latitude: 91.2\nLongitude: 10.5")["lat"] is None     # outside India


def test_stamp_ocr_roundtrip():
    pytest.importorskip("pytesseract")
    r = PS.read_stamps([cv2.cvtColor(_stamp_img(), cv2.COLOR_BGR2GRAY)])[0]
    assert r["stamp_ok"]
    assert abs(r["lat"] - 29.4106963) < 1e-3 and abs(r["lng"] - 75.8701606) < 1e-3


def test_verify_flags_real_far_stamp():
    g = cv2.cvtColor(_stamp_img(lng="75.5586803"), cv2.COLOR_BGR2GRAY)
    first = PS.read_stamps([g])[0]
    best, confirmed = PS.verify(g, first, (29.4106963, 75.35868), 200)
    assert best["stamp_ok"] and confirmed is True          # 19 km away, confirmed by a second read


def test_phash_duplicates():
    a = _stamp_img()
    b = np.roll(a, 3, axis=1)
    c = cv2.flip(a, 0)
    ca, cb, cc = PF.content(a), PF.content(b), PF.content(c)
    assert (PF.phash(ca) != PF.phash(cb)).sum() <= PL.DUP_HD
    assert PF.ssim(PF.small_gray(ca), PF.small_gray(ca)) > 0.99


def test_analyse_unreadable_and_basic(tmp_path):
    assert PL.analyse([str(tmp_path / "nope.jpg")], {})["photo_status"] == "Unavailable"
    p = tmp_path / "x_1.jpg"
    cv2.imwrite(str(p), _stamp_img())
    row = {"latitude": 29.4106963, "longitude": 75.35868, "survey_start_date": "2026-08-10", "survey_end_date": "2026-08-12"}
    old = PL._CACHE_DIR
    PL._CACHE_DIR = str(tmp_path / "cache")
    try:
        r = PL.analyse([str(p), str(p)], row)
    finally:
        PL._CACHE_DIR = old
    assert r["photo_status"] == "OK" and r["n_duplicates"] == 1
    assert r["stamp_dist_m"] and r["stamp_dist_m"] > 10000
    assert any("from app location" in x for x in r["remarks"])
    assert any("duplicate" in x.lower() for x in r["remarks"])
    assert r["photo_date"] == "14082026"
    assert any("after intimation" in x for x in r["remarks"])
    for k in ("photo_is_form", "n_form_photos", "rotated", "stamp_lat", "stamp_lng", "stamp_date", "remarks", "field_photo", "farmer_photo"):
        assert k in r
    assert r["field_photo"] in ("no crop", "cut & spread", "crop mismatch", "standing crop")
