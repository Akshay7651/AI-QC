import numpy as np

import gps_qc
from helpers import frame, good

DEG = 1 / 111_320.0  # one metre in degrees latitude


def cluster(n, surveyor="S", dmg=20, lat0=29.0, base=0, **kw):
    # n points within ~4 m of each other
    return [good(base + i, surveyor_name=surveyor, latitude=lat0 + i * DEG, longitude=76.0, total_damage_pct=dmg,
                 khasra_number=str(base + i), **kw) for i in range(n)]


def test_counts_and_qc_required():
    out = gps_qc.run(frame(cluster(5)))          # each has 4 same-surveyor neighbours (> 3)
    assert (out["Nearby_Same_Surveyor_25m"] == 4).all() and (out["Nearby_Any_Surveyor_25m"] == 4).all()
    assert (out["Suggested_Remark"] == "Same Location - QC Required").all()
    assert (out["Cluster_Size"] == 5).all() and set(out["Group_ID"]) == {"G-001"}
    assert (out["Suggest_%"] == 95).all()


def test_exactly_three_neighbours_is_ok():
    out = gps_qc.run(frame(cluster(4)))
    assert (out["Nearby_Same_Surveyor_25m"] == 3).all()
    assert (out["Suggested_Remark"] == "OK").all()


def test_low_damage_boundary():
    assert (gps_qc.run(frame(cluster(5, dmg=15)))["Suggested_Remark"] == "Same Location - Low Damage").all()
    assert (gps_qc.run(frame(cluster(5, dmg=15.5)))["Suggested_Remark"] == "Same Location - QC Required").all()
    out = gps_qc.run(frame(cluster(5, dmg=None, crop_loss_pct=90)))  # falls back to crop loss
    assert (out["Suggested_Remark"] == "Same Location - QC Required").all()
    assert (gps_qc.run(frame(cluster(5, dmg=5)))["Suggest_%"] == 75).all()


def test_other_surveyors_count_as_any_but_not_same():
    rows = cluster(5, surveyor="S") + [good(50 + i, surveyor_name=f"O{i}", latitude=29.0 + i * DEG, longitude=76.0 + 5 * DEG) for i in range(4)]
    out = gps_qc.run(frame(rows))
    assert (out["Nearby_Same_Surveyor_25m"].iloc[:5] == 4).all()
    assert (out["Nearby_Any_Surveyor_25m"].iloc[:5] >= 8).all()
    assert (out["Suggested_Remark"].iloc[5:] == "OK").all()
    assert (out["Nearby_Same_Surveyor_25m"].iloc[5:] == 0).all()


def test_beyond_radius_not_neighbours():
    rows = [good(i, surveyor_name="S", latitude=29.0 + i * 30 * DEG, longitude=76.0) for i in range(6)]
    assert (gps_qc.run(frame(rows))["Nearby_Any_Surveyor_25m"] == 0).all()
    rows = [good(i, surveyor_name="S", latitude=29.0 + i * 20 * DEG, longitude=76.0) for i in range(6)]
    assert gps_qc.run(frame(rows))["Nearby_Any_Surveyor_25m"].tolist() == [1, 2, 2, 2, 2, 1]


def test_blank_surveyor_never_same():
    out = gps_qc.run(frame(cluster(6, surveyor=None)))
    assert (out["Nearby_Same_Surveyor_25m"] == 0).all() and (out["Suggested_Remark"] == "OK").all()
    assert (out["Nearby_Any_Surveyor_25m"] == 5).all()


def test_surveyor_name_case_and_space_insensitive():
    rows = cluster(3, surveyor="Ram ") + cluster(2, surveyor="ram", base=10, lat0=29.0 + 3 * DEG)
    assert (gps_qc.run(frame(rows))["Nearby_Same_Surveyor_25m"] == 4).all()


def test_multiple_records_same_field():
    rows = [good(i, khasra_number="99", patwar_circle="P1", latitude=29.0 + i * 0.01) for i in range(4)]
    out = gps_qc.run(frame(rows))
    assert (out["Records_On_Same_Field"] == 4).all()
    assert (out["Suggested_Remark"] == "Review - Multiple Records").all() and (out["Suggest_%"] == 65).all()
    out = gps_qc.run(frame(rows[:3]))
    assert (out["Suggested_Remark"] == "OK").all()      # exactly 3 is not flagged
    rows = [dict(r, patwar_circle=f"P{i}") for i, r in enumerate(rows)]
    assert (gps_qc.run(frame(rows))["Records_On_Same_Field"] == 1).all()


def test_same_location_wins_over_multiple_records():
    rows = cluster(5, patwar_circle="P1")
    rows = [dict(r, khasra_number="1") for r in rows]
    assert (gps_qc.run(frame(rows))["Suggested_Remark"] == "Same Location - QC Required").all()


def test_invalid_gps_and_empty_frame():
    rows = [good(0, latitude=None, longitude=None), good(1, latitude=None, longitude=None), good(2)]
    out = gps_qc.run(frame(rows))
    assert list(out["Nearby_Any_Surveyor_25m"]) == [0, 0, 0] and out["Group_ID"].tolist() == ["", "", ""]
    assert len(gps_qc.run(frame([good(0)]).iloc[0:0])) == 0


def test_non_default_index_and_group_ids():
    rows = cluster(5, base=0) + cluster(5, base=20, lat0=30.0, surveyor="T")
    df = frame(rows)
    df.index = np.arange(100, 110)
    out = gps_qc.run(df)
    assert list(out.index) == list(range(100, 110))
    assert out["Group_ID"].tolist() == ["G-001"] * 5 + ["G-002"] * 5
