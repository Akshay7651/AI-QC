import pandas as pd

import data_qc
from helpers import frame, good


def flags(rows):
    return list(data_qc.run(frame(rows)))


def test_clean_row_has_no_flags():
    assert flags([good(0), good(1)]) == ["", ""]


def test_missing_required():
    f = flags([good(0, farmer_name=None, latitude=None, crop_name="  ")])[0]
    assert "Missing: farmer_name" in f and "Missing: latitude" in f and "Missing: crop_name" in f


def test_range_and_total_damage():
    f = flags([good(0, affected_area_pct=120), good(1, crop_loss_pct=-1), good(2, total_damage_pct=90)])
    assert "affected_area_pct out of range" in f[0]
    assert "crop_loss_pct out of range" in f[1]
    assert "Total damage != affected x loss" in f[2]


def test_gps_bounds():
    f = flags([good(0, latitude=0.0, longitude=0.0), good(1, latitude=51.0)])
    assert "GPS out of India bounds" in f[0] and "GPS out of India bounds" in f[1]


def test_dates():
    f = flags([good(0, survey_start_date="garbage"), good(1, survey_start_date="2026-09-09", survey_end_date="2026-09-01"),
               good(2, survey_start_date="2999-01-01", survey_end_date="2999-01-02")])
    assert "Unparseable survey_start_date" in f[0]
    assert "Start date after end date" in f[1]
    assert "Survey date in future" in f[2]


def test_season_window_needs_ten_rows():
    rows = [good(i) for i in range(12)]
    rows.append(good(12, survey_start_date="2020-01-01", survey_end_date="2020-01-02"))
    f = flags(rows)
    assert "outside season window" in f[12] and all("season" not in x for x in f[:12])
    assert all("season" not in x for x in flags(rows[:5] + rows[-1:]))


def test_duplicates_and_same_field():
    f = flags([good(0, docket_id="X"), good(1, docket_id="X"),
               good(2, docket_id="A", khasra_number="7", village="VV"), good(3, docket_id="B", khasra_number="7", village="VV"),
               good(4, docket_id="C", khasra_number="7", village="OTHER")])
    assert "Duplicate record" in f[0] and "Duplicate record" in f[1]
    assert "Possible duplicate field" in f[2] and "Possible duplicate field" in f[3]
    assert "Possible duplicate field" not in f[4] and "Duplicate record" not in f[2]


def test_mobile():
    f = flags([good(0, surveyor_mobile="12345"), good(1, surveyor_mobile="9123456789.0"), good(2, surveyor_mobile="+91 91234 56789"),
               good(3, surveyor_mobile=None)])
    assert "Invalid mobile number" in f[0]
    assert f[1] == "" and f[2] == "" and f[3] == ""


def test_empty_and_all_null_frames():
    assert len(data_qc.run(frame([good(0)]).iloc[0:0])) == 0
    f = flags([{"docket_id": None}])
    assert "Missing: docket_id" in f[0]


def test_iso_dates_not_day_month_swapped():
    import common
    d = common.parse_dates(["2026-09-01", "01/09/2026", "13-09-2026", None, "junk", "2026-09-01 10:00:00"])
    assert [x.strftime("%Y-%m-%d") if pd.notna(x) else None for x in d] == [
        "2026-09-01", "2026-09-01", "2026-09-13", None, None, "2026-09-01"]
    assert flags([good(0, survey_start_date="2026-09-05", survey_end_date="2026-09-06")]) == [""]
