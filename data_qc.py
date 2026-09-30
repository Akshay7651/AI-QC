"""Rule-based data validation. No API calls."""
import numpy as np
import pandas as pd

import config as C
from common import parse_dates

REQUIRED = ["docket_id", "farmer_name", "crop_name", "khasra_number",
            "latitude", "longitude", "affected_area_pct", "crop_loss_pct"]


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip() == "")


def run(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    flags = [[] for _ in range(n)]

    def add(mask, text):
        for i in np.flatnonzero(np.asarray(mask)):
            flags[i].append(text)

    for f in REQUIRED:
        add(_blank(df[f]), f"Missing: {f}")

    for f in ["affected_area_pct", "crop_loss_pct", "total_damage_pct"]:
        v = df[f]
        add(v.notna() & ((v < 0) | (v > 100)), f"{f} out of range 0-100")
    expected = df["affected_area_pct"] * df["crop_loss_pct"] / 100
    add(df["total_damage_pct"].notna() & expected.notna()
        & ((df["total_damage_pct"] - expected).abs() > C.TOTAL_DAMAGE_TOLERANCE_PCT),
        "Total damage != affected x loss")

    lat, lng = df["latitude"], df["longitude"]
    has_gps = lat.notna() & lng.notna()
    add(has_gps & ~(lat.between(C.INDIA_LAT_MIN, C.INDIA_LAT_MAX) & lng.between(C.INDIA_LNG_MIN, C.INDIA_LNG_MAX)),
        "GPS out of India bounds")

    start = parse_dates(df["survey_start_date"])
    end = parse_dates(df["survey_end_date"])
    add(~_blank(df["survey_start_date"]) & start.isna(), "Unparseable survey_start_date")
    add(~_blank(df["survey_end_date"]) & end.isna(), "Unparseable survey_end_date")
    add(start.notna() & end.notna() & (start > end), "Start date after end date")
    # Season window is derived from the data itself (median date) - no hardcoded season.
    for name, s in (("start", start), ("end", end)):
        if s.notna().sum() >= 10:
            med = s.median()
            add(s.notna() & ((s - med).abs() > pd.Timedelta(days=C.SEASON_WINDOW_DAYS)), f"Survey {name} date outside season window")
    add(start.notna() & (start > pd.Timestamp.now() + pd.Timedelta(days=1)), "Survey date in future")

    dup = df["docket_id"].notna() & df.duplicated("docket_id", keep=False)
    add(dup, "Duplicate record")
    fld = ["khasra_number", "division_number", "village"]
    ok = df[fld].notna().all(axis=1)
    multi = df[ok].groupby(fld)["docket_id"].transform("nunique") > 1
    add(ok & multi.reindex(df.index, fill_value=False), "Possible duplicate field")

    mob = df["surveyor_mobile"].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True).str[-10:]
    add(~_blank(df["surveyor_mobile"]) & ~mob.str.fullmatch(r"[6-9]\d{9}"), "Invalid mobile number")

    return pd.Series([", ".join(f) for f in flags], index=df.index, name="Data_QC_Flags")
