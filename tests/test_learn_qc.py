"""Tests for learn_qc.py on synthetic data only (no real farmer data)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C  # noqa: E402
import learn_qc  # noqa: E402
from column_map import DEFAULT_ORDER  # noqa: E402


def synth(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({c: [None] * n for c in DEFAULT_ORDER})
    d["docket_id"] = [f"D{i:06d}" for i in range(n)]
    d["farmer_name"] = "x"
    d["crop_name"] = rng.choice(["A", "B"], n)
    d["calamity_type"] = "Flood"
    d["district"] = rng.choice(["d1", "d2"], n)
    d["village"] = rng.choice([f"v{i}" for i in range(15)], n)
    d["surveyor_name"] = [f"S{i % 30}" for i in range(n)]
    d["survey_start_date"] = pd.Timestamp("2026-09-01") + pd.to_timedelta(rng.integers(0, 10, n), "D")
    d["survey_start_date"] = d["survey_start_date"].dt.strftime("%Y-%m-%d")
    d["survey_end_date"] = "2026-09-20"
    d["khasra_number"] = rng.integers(1, 5000, n).astype(str)
    d["farm_area"] = rng.uniform(0.2, 2, n)
    d["affected_area_pct"] = rng.integers(10, 100, n).astype(float)
    d["crop_loss_pct"] = rng.integers(3, 100, n).astype(float)
    d["total_damage_pct"] = d["affected_area_pct"] * d["crop_loss_pct"] / 100
    d["latitude"] = 29 + rng.random(n) * 0.5 + rng.random(n) * 1e-6
    d["longitude"] = 76 + rng.random(n) * 0.5 + rng.random(n) * 1e-6
    # planted bad surveyor S0: same loss / same spot / round numbers
    bad = d.index[d.surveyor_name == "S0"]
    d.loc[bad, ["affected_area_pct", "crop_loss_pct", "total_damage_pct"]] = [60.0, 60.0, 36.0]
    d.loc[bad, ["latitude", "longitude"]] = [29.1, 76.1]
    return d, set(bad)


def test_risk_flags_planted_surveyor_and_is_deterministic():
    d, bad = synth()
    a = learn_qc.add_risk(d)
    b = learn_qc.add_risk(d)
    assert a["Risk_Score"].equals(b["Risk_Score"]) and a["Risk_Reasons"].equals(b["Risk_Reasons"])
    assert a["Risk_Score"].between(0, 100).all()
    assert a.loc[list(bad), "Risk_Score"].mean() > a.drop(index=list(bad))["Risk_Score"].mean() + 25
    assert (a.loc[list(bad), "Risk_Reasons"] != "").mean() > 0.9
    assert len(a) == len(d) and "Risk_Score" not in d  # input untouched


def test_risk_handles_missing_and_empty():
    d, _ = synth(300)
    d.loc[:50, ["latitude", "farm_area", "crop_loss_pct"]] = np.nan
    d["surveyor_name"] = None
    out = learn_qc.add_risk(d)
    assert out["Risk_Score"].notna().all()
    assert len(learn_qc.add_risk(d.iloc[0:0])) == 0


def labelled(n=1200):
    d, _ = synth(n)
    rng = np.random.default_rng(1)
    # Human "Mismatch" more likely when loss is very high; status depends on district
    p = 1 / (1 + np.exp(-(d.crop_loss_pct - 80) / 4))
    d[C.COL_MATCH] = np.where(rng.random(n) < p, "Mismatch", "Match")
    d[C.COL_FORM_STATUS] = np.where(d.district == "d1", "correct", "incomplete")
    d[C.COL_FIELD_PHOTO] = None  # unlabelled target -> must be skipped
    d.loc[:20, C.COL_FARMER_PHOTO] = "Yes"  # too few labels -> must be skipped
    return d


def test_train_predict_roundtrip(tmp_path, monkeypatch, capsys):
    d = labelled()
    monkeypatch.setattr(learn_qc, "_load", lambda p: d.copy())
    model = str(tmp_path / "m.joblib")
    assert learn_qc.train("x.xlsx", model) == 0
    txt = capsys.readouterr().out
    assert "[Match]" in txt and "[FormStatus]" in txt and "SKIPPED FieldPhoto" in txt and "SKIPPED FarmerPhoto" in txt
    out = learn_qc.add_predictions(d.drop(columns=[C.COL_MATCH, C.COL_FORM_STATUS]), model)
    assert set(out["ML_FormStatus"]) <= {"correct", "incomplete"}
    assert (out["ML_FormStatus"] == np.where(d.district == "d1", "correct", "incomplete")).mean() > 0.95
    assert out["ML_Match_Conf"].between(0, 1).all()


def test_train_refuses_without_labels(tmp_path, monkeypatch, capsys):
    d, _ = synth(500)
    for c in learn_qc.TARGETS:
        d[c] = None
    monkeypatch.setattr(learn_qc, "_load", lambda p: d.copy())
    model = tmp_path / "m.joblib"
    assert learn_qc.train("x.xlsx", str(model)) == 2
    assert not model.exists()
    assert "too few human-filled QC labels" in capsys.readouterr().out


def test_feedback_per_field(monkeypatch, capsys):
    d = labelled(400)
    ai = d.copy()
    hu = d.copy()
    ai[C.COL_MATCH] = "Match"
    hu[C.COL_MATCH] = ["Match"] * 300 + ["Mismatch"] * 100
    ai[C.COL_FORM_LOSS], hu[C.COL_FORM_LOSS] = "50", "52"
    monkeypatch.setattr(learn_qc, "_load", lambda p: ai.copy() if p == "ai" else hu.copy())
    res, cf = learn_qc.feedback("ai", "hu")
    r = res.set_index("field")
    assert r.loc[C.COL_MATCH, "agreement_%"] == 75.0
    assert r.loc[C.COL_FORM_LOSS, "agreement_%"] == 100.0  # within tolerance
    assert cf.iloc[0]["ai_said"] == "match" and cf.iloc[0]["human_said"] == "mismatch"
