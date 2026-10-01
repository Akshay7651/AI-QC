import datetime
import os

import numpy as np
import pytest

import date_reader as DR

FIX = os.path.join(os.path.dirname(__file__), "fixtures_dates")


@pytest.mark.parametrize("text,exp", [
    ("01-05-2026", "01052026"), ("01-05-26", "01052026"), ("1-6-26", "01062026"), ("10-06-26", "10062026"),
    ("1-06-2026", "01062026"), ("16-8-26", "16082026"), ("10-08-2026", "10082026"),
    ("01052026", "01052026"), ("010526", "01052026"), ("1626", "01062026"), ("10626", None),   # 10626: 1/06/26 or 10/6/26
    ("32-05-26", None), ("10-13-26", None), ("10-06-2031", None), ("", None), ("--", None),
])
def test_parse(text, exp):
    c = DR.candidates(text, DR.WINDOWS["loss"] if exp is None or exp[2:4] >= "06" else DR.WINDOWS["sow"])
    got = c[0].strftime("%d%m%Y") if len(c) == 1 else None
    if text == "10626":
        assert len(c) == 2       # ambiguous -> never asserted
    else:
        assert got == exp


def test_window_sow_vs_loss():
    assert DR.candidates("10-02-26", DR.WINDOWS["sow"]) == []       # sowing in February: implausible
    assert DR.candidates("10-09-26", DR.WINDOWS["loss"]) == [datetime.date(2026, 9, 10)]


def test_blank_and_tiny():
    r = DR.read_date(np.full((40, 200), 230, np.uint8))
    assert r["date"] is None
    assert DR.read_date(np.zeros((2, 2), np.uint8))["date"] is None


@pytest.mark.skipif(not os.path.exists(DR.MODEL_PATH) or not os.path.isdir(FIX), reason="date model / fixtures missing")
def test_fixture_crops():
    import cv2
    import csv
    rows = list(csv.DictReader(open(os.path.join(FIX, "labels.csv"))))
    assert rows
    ok = 0
    for r in rows:
        g = cv2.imread(os.path.join(FIX, r["file"]), 0)
        out = DR.read_date(g, r["field"])
        ok += out["date"] == r["date"]
        assert out["date"] in (r["date"], None) or out["conf"] < 0.99
    assert ok >= 0.8 * len(rows)


@pytest.mark.skipif(not os.path.exists(DR.MODEL_PATH), reason="date model missing")
def test_speed():
    import time
    g = (np.random.rand(60, 300) * 255).astype(np.uint8)
    DR.read_date(g)
    t = time.time()
    for _ in range(5):
        DR.read_date(g)
    assert (time.time() - t) / 5 < 0.1
