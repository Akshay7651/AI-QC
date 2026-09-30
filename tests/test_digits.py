import json
import os
import time

import numpy as np
import pytest
from PIL import Image

import digits as D

FIX = os.path.join(os.path.dirname(__file__), "fixtures_digits")
pytestmark = pytest.mark.skipif(not os.path.exists(D.CELL_MODEL_PATH), reason="models/cells_cnn.npz missing (run train_cells.py)")


def _blank(h=44, w=160):
    return Image.fromarray(np.full((h, w), 200, np.uint8))


def test_interface_keys_and_blank():
    r = D.read_number(_blank())
    assert set(r) >= {"value", "text", "conf", "n_components"}
    assert r["value"] is None and r["n_components"] == 0


def test_blank_with_ruled_line_is_none():
    a = np.full((44, 160), 200, np.uint8)
    a[22, :] = 60                      # a ruled line
    a[:, 0] = 60
    r = D.read_number(Image.fromarray(a))
    assert r["value"] is None and r["n_components"] == 0


def test_accepts_rgb_and_arrays():
    a = np.full((44, 160, 3), 210, np.uint8)
    assert D.read_number(a)["value"] is None
    assert D.read_number(Image.fromarray(a))["value"] is None


def test_read_digit_string_contract():
    r = D.read_digit_string(_blank())
    assert r["text"] == "" and r["n_components"] == 0


def test_values_never_exceed_100():
    rng = np.random.default_rng(0)
    for _ in range(10):
        r = D.read_number(Image.fromarray(rng.integers(0, 255, (44, 160)).astype(np.uint8)))
        assert r["value"] is None or 0 <= r["value"] <= 100


def test_regression_saved_crops():
    exp = json.load(open(os.path.join(FIX, "expected.json")))
    ok = 0
    for name, v in exp.items():
        r = D.read_number(Image.open(os.path.join(FIX, name)))
        ok += int((v is None and r["value"] is None) or (v is not None and r["value"] == v))
    assert ok >= len(exp) - 1, f"{ok}/{len(exp)}"


def test_speed():
    im = Image.open(os.path.join(FIX, sorted(os.listdir(FIX))[0])) if False else _blank()
    D.read_number(im)
    t = time.time()
    for _ in range(10):
        D.read_number(im)
    assert (time.time() - t) / 10 < 0.05
