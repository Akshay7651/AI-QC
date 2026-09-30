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
