"""Tests for the Proforma-3 reader (form_p3.py geometry + form_reader.py)."""
import glob
import os

import numpy as np
import pytest
from PIL import Image

import form_p3 as P
import form_reader as R

KEYS = ["is_proforma3", "quality", "form_no", "form_no_conf", "po_id", "po_id_matches", "form_area", "form_loss",
        "row_area", "row_loss", "total_row_blank", "sow_date", "loss_date", "intimation_date", "inspection_date",
        "farmer_signed", "company_signed", "worker_signed", "officer_signed", "officer_stamp_only",
        "overwrite_suspected", "confidence", "field_conf", "notes"]

FORMS = sorted(glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "forms", "*.jpg")))


def test_assemble_date_variants():
    assert R._assemble_date(["10", "08", "2026"], 0.9)[0] == "10082026"
    assert R._assemble_date(["1", "8", "26"], 0.9)[0] == "01082026"
    assert R._assemble_date(["10082026"], 0.9)[0] == "10082026"
    assert R._assemble_date(["100826"], 0.9)[0] == "10082026"
    assert R._assemble_date(["31", "02", "2026"], 0.9)[0] is None       # impossible date
    assert R._assemble_date(["5"], 0.9)[0] is None


def test_formno_normalisation():
    assert R._find_formno("HR0126175631") == "HR0126175631"
    assert R._find_formno("2 HR0126175631") == "HR0126175631"
    assert R._find_formno("HR126175631") == "HR0126175631"      # dropped leading zero
    assert R._find_formno("4R0126175631") == "HR0126175631"
    assert R._find_formno("HR01261756") is None


def test_blank_image_is_not_a_form():
    out = R.read_form(Image.fromarray(np.full((800, 600, 3), 255, np.uint8)))
    assert out["is_proforma3"] is False
    for k in KEYS:
        assert k in out


def test_photo_of_random_noise_is_not_a_form():
    rng = np.random.RandomState(0)
    out = R.read_form(Image.fromarray(rng.randint(0, 255, (900, 700, 3), dtype=np.uint8)))
    assert out["is_proforma3"] is False


@pytest.mark.skipif(not FORMS, reason="no data/forms")
def test_real_form_keys_and_formno():
    out = R.read_form(FORMS[0], os.path.basename(FORMS[0])[:-4])
    for k in KEYS:
        assert k in out
    assert out["is_proforma3"] is True
    if out["form_no"]:
        assert out["form_no"].startswith("HR0126") and len(out["form_no"]) == 12


@pytest.mark.skipif(not FORMS, reason="no data/forms")
def test_rotation_invariance():
    f = FORMS[1]
    base = R.read_form(f)
    img = Image.open(f)
    for ang in (90, 180, 270):
        rot = R.read_form(img.rotate(ang, expand=True))
        assert rot["is_proforma3"] is True
        assert rot["form_no"] == base["form_no"]


@pytest.mark.skipif(not FORMS, reason="no data/forms")
def test_layout_boxes_inside_image():
    lay = P.analyse_layout(P.load_image(FORMS[2]))
    if lay["ok"]:
        h, w = lay["rgb"].shape[:2]
        for k, (x0, y0, x1, y1) in lay["boxes"].items():
            assert x1 > x0 and y1 > y0, k


def _fake_probs(seq):
    """seq of chars '0'-'9' or '/' -> (probs (N,13), comps) as digits.py would hand back"""
    probs, comps = [], []
    for i, ch in enumerate(seq):
        v = np.full(13, 0.001)
        v[int(ch) if ch.isdigit() else 11] = 0.98
        probs.append(v)
        tall = 1.6 if ch == "/" else 1.0
        comps.append({"x0": i * 20, "x1": i * 20 + (8 if ch == "/" else 14), "y0": 0, "y1": int(30 * tall), "dot": False})
    return np.array(probs), comps


@pytest.mark.parametrize("txt,want", [("10/08/2026", "10082026"), ("9/8/26", "09082026"), ("10-08-26", "10082026"), ("10082026", "10082026")])
def test_parse_date_from_component_classes(monkeypatch, txt, want):
    txt2 = txt.replace("-", "/")
    monkeypatch.setattr(R, "_digit_probs", lambda crop: _fake_probs(txt2))
    monkeypatch.setattr(P, "ink_frac", lambda crop, inset=0.06, drop_strokes=True: 0.1)
    d, conf, note = R.parse_date(np.zeros((40, 200, 3), np.uint8))
    assert d == want


def test_po_id_verification_with_fake_digits(monkeypatch):
    docket = "040106260000636479"
    monkeypatch.setattr(R, "_digit_probs", lambda crop: _fake_probs(docket))
    monkeypatch.setattr(P, "ink_frac", lambda crop, inset=0.06, drop_strokes=True: 0.1)
    po, conf, match, info = R.read_po_id(np.zeros((40, 400, 3), np.uint8), docket)
    assert po == docket and match is True
    po, conf, match, info = R.read_po_id(np.zeros((40, 400, 3), np.uint8), "040106260000636478")
    assert match is False
