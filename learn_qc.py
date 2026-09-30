#!/usr/bin/env python3
"""Learning layer for the CLAP QC pipeline.  No API calls, deterministic (fixed seeds).

A) UNSUPERVISED risk model - works on any export today, needs no human labels.
       add_risk(df) -> df + Risk_Score (0-100), Risk_Reasons
   Per-row features (loss/area/damage, farm area, GPS-cluster counts from gps_qc, per-surveyor behaviour
   such as identical / round-number / copy-pasted loss values, records per day, coordinate precision,
   village-level robust z-scores) feed an IsolationForest + robust-z ensemble.  The score is a RELATIVE
   review-priority (percentile-style), not a probability of fraud.  Reasons are rule-based and explainable.

B) SUPERVISED trainer - only useful once humans have filled the QC columns.
       python learn_qc.py train   --input qc_reviewed.xlsx [--ai-output qc_ai.xlsx] --model output/model.joblib
       python learn_qc.py predict --input new.xlsx --model output/model.joblib --output new_ML.xlsx
       python learn_qc.py feedback --ai qc_ai.xlsx --human qc_reviewed.xlsx --output feedback.xlsx
   IMPORTANT: B does NOT train an OCR / vision reader.  It learns *reviewer patterns* (which rows humans mark
   Mismatch / incomplete / bad photo) from tabular signals, so it can prioritise or pre-fill.  `feedback`
   measures where the AI's per-field output disagrees with human corrections so the OCR/photo prompts in
   pdf_qc.py / photo_qc.py can be tuned by hand.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

SEED = 0
MIN_SURVEYOR_ROWS = 20   # surveyors with fewer rows are not compared with peers
MIN_GROUP_ROWS = 10      # village / crop groups smaller than this get z = 0
MIN_LABELLED = 200       # per target, for supervised training
RISK_OUT = ["Risk_Score", "Risk_Reasons"]

# label columns (config names) -> short name used in ML_<short> output columns
TARGETS = {
    C.COL_MATCH: "Match",
    C.COL_FORM_STATUS: "FormStatus",
    C.COL_FIELD_PHOTO: "FieldPhoto",
    C.COL_FARMER_PHOTO: "FarmerPhoto",
}
NEEDS_CORRECTION = "Needs_Correction"
COMPARE_COLS = [C.COL_FORM_AREA, C.COL_FORM_LOSS, C.COL_MATCH, C.COL_PHOTO_DATE, C.COL_FIELD_PHOTO,
                C.COL_SURVEYOR_SIG, C.COL_FARMER_SIG, C.COL_GOVT_SIG, C.COL_FORM_STATUS,
                C.COL_FARMER_PHOTO, C.COL_PHOTO_LOSS]
NUMERIC_COMPARE = {C.COL_FORM_AREA: C.AREA_MATCH_TOLERANCE_PCT, C.COL_FORM_LOSS: C.LOSS_MATCH_TOLERANCE_PCT}


# ----------------------------------------------------------------------------------------------- helpers
def _rz(x: pd.Series, key=None, min_n=MIN_GROUP_ROWS) -> pd.Series:
    """Robust z-score (median/MAD, falling back to std) globally or within `key` groups; clipped to +-10."""
    x = pd.to_numeric(x, errors="coerce")
    if key is None:
        key = pd.Series(0, index=x.index)
    g = x.groupby(key)
    med = g.transform("median")
    mad = (x - med).abs().groupby(key).transform("median") * 1.4826
    sd = g.transform("std")
    n = g.transform("count")
    scale = mad.where(mad > 0, sd)
    z = (x - med) / scale.where(scale > 0)
    z = z.where(n >= min_n)
    return z.fillna(0.0).clip(-10, 10)


def _decimals(s: pd.Series) -> np.ndarray:
    def d(v):
        if v != v:
            return 0
        t = repr(float(v))
        return len(t.split(".")[1]) if "." in t and "e" not in t else 0
    return s.map(d).to_numpy(int)


def _surveyor_key(df: pd.DataFrame) -> pd.Series:
    k = df["surveyor_name"].astype("object").where(df["surveyor_name"].notna(), None)
    if "surveyor_mobile" in df:
        k = k.where(k.notna(), df["surveyor_mobile"])
    return k.fillna("").astype(str).str.strip().str.lower()


def _area_key(df: pd.DataFrame) -> pd.Series:
    """Smallest reliable geographic unit present (village, else patwar circle, else tehsil, else district)."""
    for c in ("village", "patwar_circle", "tehsil", "district"):
        if c in df and df[c].notna().mean() > 0.5:
            return df[c].fillna("").astype(str).str.strip().str.lower()
    return pd.Series("", index=df.index)


def _gps_features(df: pd.DataFrame) -> pd.DataFrame:
    need = ["Nearby_Same_Surveyor_25m", "Nearby_Any_Surveyor_25m", "Records_On_Same_Field", "Cluster_Size"]
    if all(c in df for c in need):
        return df[need].apply(pd.to_numeric, errors="coerce").fillna(0)
    import gps_qc
    return gps_qc.run(df)[need].apply(pd.to_numeric, errors="coerce").fillna(0)


def _peer_z(stat: pd.Series, n: pd.Series) -> pd.Series:
    """Robust z of a per-surveyor statistic against surveyors that have enough rows."""
    ok = n >= MIN_SURVEYOR_ROWS
    z = pd.Series(0.0, index=stat.index)
    if ok.sum() >= 5:
        z[ok] = _rz(stat[ok], min_n=5)
    return z


# ----------------------------------------------------------------------------------------------- features
def compute_features(df: pd.DataFrame):
    """Return (numeric feature DataFrame, list of (weight ndarray in 0..1, text ndarray) reason rules)."""
    n = len(df)
    idx = df.index
    loss = pd.to_numeric(df["crop_loss_pct"], errors="coerce")
    aff = pd.to_numeric(df["affected_area_pct"], errors="coerce")
    tot = pd.to_numeric(df["total_damage_pct"], errors="coerce")
    farm = pd.to_numeric(df["farm_area"], errors="coerce")
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lng = pd.to_numeric(df["longitude"], errors="coerce")
    surv = _surveyor_key(df)
    area = _area_key(df)
    crop = df["crop_name"].fillna("").astype(str).str.strip().str.lower() if "crop_name" in df else pd.Series("", index=idx)
    gps = _gps_features(df)
    F = pd.DataFrame(index=idx)
    F["crop_loss"], F["affected_area"], F["total_damage"] = loss, aff, tot
    F["log_farm_area"] = np.log1p(farm.clip(lower=0))
    F["nearby_same_surveyor"] = gps["Nearby_Same_Surveyor_25m"].to_numpy()
    F["nearby_any_surveyor"] = gps["Nearby_Any_Surveyor_25m"].to_numpy()
    F["records_same_field"] = gps["Records_On_Same_Field"].to_numpy()
    F["cluster_size"] = gps["Cluster_Size"].to_numpy()

    # coordinates: precision and exact duplicates
    dec = np.where(lat.notna() & lng.notna(), np.minimum(_decimals(lat), _decimals(lng)), 0)
    F["coord_decimals"] = dec
    ck = lat.round(7).astype(str) + "," + lng.round(7).astype(str)
    F["coord_exact_dups"] = (ck.groupby(ck).transform("size") - 1).where(lat.notna() & lng.notna(), 0).to_numpy()

    # consistency
    exp = aff * loss / 100
    F["damage_mismatch"] = (tot - exp).abs()
    F["loss_zero_area_pos"] = ((loss == 0) & (aff > 0)).astype(int)
    F["loss_pos_area_zero"] = ((loss > 0) & (aff == 0)).astype(int)
    sd = pd.to_datetime(df["survey_start_date"], errors="coerce", format="mixed", dayfirst=True)
    ed = pd.to_datetime(df["survey_end_date"], errors="coerce", format="mixed", dayfirst=True)
    F["survey_span_days"] = (ed - sd).dt.days

    # surveyor / day volume
    day = sd.dt.strftime("%Y-%m-%d").fillna("")
    sday = surv + "|" + day
    F["surveyor_day_records"] = sday.groupby(sday).transform("size").to_numpy()
    sd_cnt = pd.Series(F["surveyor_day_records"].to_numpy(), index=idx).groupby(sday).first()
    F["surveyor_day_z"] = sday.map(_rz(np.log1p(sd_cnt), min_n=5)).to_numpy(float)

    # per-surveyor behaviour
    nz = loss.notna() & (loss > 0)
    tmp = pd.DataFrame({"s": surv, "loss": loss, "nz": nz, "aff": aff, "tot": tot, "farm": farm,
                        "round5": (loss % 5 == 0) & nz, "lowprec": dec <= 4,
                        "zero": loss == 0, "day": day})
    tmp["tuple"] = (aff.astype(str) + "|" + loss.astype(str) + "|" + tot.astype(str) + "|" + farm.astype(str))
    tmp["tuple_n"] = tmp.groupby(["s", "tuple"])["tuple"].transform("size")
    tmp["tuple_dup"] = (tmp["tuple_n"] > 1) & nz
    nrows = tmp.groupby("s")["loss"].transform("size")
    nnz = tmp.groupby("s")["nz"].transform("sum")
    mode_nz = tmp[nz].groupby("s")["loss"].agg(lambda v: v.value_counts().iloc[0])
    grp = tmp.groupby("s")
    S = pd.DataFrame({
        "n": grp.size(),
        "n_nz": grp["nz"].sum(),
        "mode_share": mode_nz.reindex(grp.size().index) / grp["nz"].sum().clip(lower=1),
        "round5_share": grp["round5"].sum() / grp["nz"].sum().clip(lower=1),
        "tuple_dup_share": grp["tuple_dup"].sum() / grp["nz"].sum().clip(lower=1),
        "zero_share": grp["zero"].mean(),
        "lowprec_share": grp["lowprec"].mean(),
        "mean_loss": grp["loss"].mean(),
        "per_day": grp["day"].apply(lambda d: len(d) / max(d.nunique(), 1)),
    })
    S["mode_share"] = S["mode_share"].fillna(0)
    # digit preference: share of non-zero losses ending in 0 or 5 is round5; also max last-digit share
    ld = (loss[nz] % 10).astype(int)
    dp = ld.groupby(surv[nz]).agg(lambda v: v.value_counts(normalize=True).iloc[0])
    S["digit_pref"] = dp.reindex(S.index).fillna(0)
    z = {c: _peer_z(S[c], S["n"]) for c in ["mode_share", "round5_share", "tuple_dup_share", "zero_share",
                                               "lowprec_share", "digit_pref", "per_day"]}
    # surveyor mean loss deviation vs village peers
    vz = _rz(loss.where(nz), area)
    S["village_dev"] = vz.groupby(surv).mean().reindex(S.index).fillna(0)
    z["village_dev"] = _peer_z(S["village_dev"], S["n"])
    for c, v in z.items():
        F[f"surv_{c}"] = surv.map(S[c]).to_numpy(float)
        F[f"surv_{c}_z"] = surv.map(v).to_numpy(float)
    F["surv_rows"] = surv.map(S["n"]).to_numpy(float)
    F["row_tuple_repeat"] = np.where(nz, tmp["tuple_n"].to_numpy(), 1)

    # village / crop-level z-scores
    F["village_loss_z"] = _rz(loss, area)
    F["village_aff_z"] = _rz(aff, area)
    F["village_damage_z"] = _rz(tot, area)
    F["crop_farm_area_z"] = _rz(F["log_farm_area"], crop)

    # ---------------- explainable reasons: (strength 0..1, text) ----------------
    rules = []

    def rule(strength, mask, text_fn):
        s = np.where(np.asarray(mask, bool), np.clip(np.nan_to_num(np.asarray(strength, float)), 0, 1), 0.0)
        t = np.array([text_fn(i) if s[i] > 0 else "" for i in range(n)], dtype=object) if s.any() else np.full(n, "", object)
        rules.append((s, t))

    # Count-type signals are judged against the dataset itself (log-scale robust z), because some exports
    # legitimately have very dense GPS stamping; a fixed count would flag most rows.
    def lz(col):
        return _rz(np.log1p(F[col])).to_numpy()
    ns, nsz = F["nearby_same_surveyor"].to_numpy(), lz("nearby_same_surveyor")
    nsm = float(np.median(ns))
    rule(nsz / 6, (ns > C.GPS_SAME_SURVEYOR_MIN) & (nsz > 2.5),
         lambda i: f"{int(ns[i])} same-surveyor records within {C.GPS_PROXIMITY_RADIUS_M} m (dataset median {nsm:.0f})")
    rf, rfz = F["records_same_field"].to_numpy(), lz("records_same_field")
    rule(rfz / 6, (rf > C.SAME_FIELD_FLAG_COUNT) & (rfz > 2.5), lambda i: f"{int(rf[i])} records share this survey number")
    ed_, edz = F["coord_exact_dups"].to_numpy(), lz("coord_exact_dups")
    rule(edz / 6, (ed_ >= 2) & (edz > 2.5), lambda i: f"identical GPS coordinates on {int(ed_[i]) + 1} records")
    sdr, sdz = F["surveyor_day_records"].to_numpy(), F["surveyor_day_z"].to_numpy()
    sdm = float(np.median(sdr))
    rule(sdz / 6, (sdz > 3.5) & (sdr >= 5),
         lambda i: f"surveyor logged {int(sdr[i])} records on this day (typical {sdm:.0f})")
    tr, tz = F["row_tuple_repeat"].to_numpy(), F["surv_tuple_dup_share_z"].to_numpy()
    tds = F["surv_tuple_dup_share"].to_numpy()
    rule(np.maximum(tz, 0) / 8, (tr >= 5) & (tz > 2.5),
         lambda i: f"same area/loss/damage/farm-area values repeated {int(tr[i])}x by this surveyor (copy-paste pattern)")
    mz, ms = F["surv_mode_share_z"].to_numpy(), F["surv_mode_share"].to_numpy()
    rule(mz / 8, (mz > 3.5) & nz.to_numpy(),
         lambda i: f"surveyor enters one loss value on {ms[i]:.0%} of non-zero forms (peers much lower)")
    rz_, rs = F["surv_round5_share_z"].to_numpy(), F["surv_round5_share"].to_numpy()
    rule(rz_ / 8, (rz_ > 3.5) & nz.to_numpy(),
         lambda i: f"surveyor: {rs[i]:.0%} of losses are round numbers (digit preference)")
    dz, dpv = F["surv_digit_pref_z"].to_numpy(), F["surv_digit_pref"].to_numpy()
    rule(dz / 8, (dz > 3.5) & nz.to_numpy(), lambda i: f"surveyor: one last-digit in {dpv[i]:.0%} of losses")
    zz, zs = F["surv_zero_share_z"].to_numpy(), F["surv_zero_share"].to_numpy()
    rule(np.abs(zz) / 8, (np.abs(zz) > 3.5) & (loss == 0).to_numpy(),
         lambda i: f"surveyor reports zero loss on {zs[i]:.0%} of rows (peers differ)")
    pz = F["surv_per_day_z"].to_numpy()
    pd_ = F["surv_per_day"].to_numpy()
    rule(pz / 8, pz > 3.5, lambda i: f"surveyor averages {pd_[i]:.0f} records per active day")
    lp = F["surv_lowprec_share_z"].to_numpy()
    dec_ = F["coord_decimals"].to_numpy()
    rule(np.maximum(lp, (5 - dec_) / 3) / 8 + (dec_ <= 3) * 0.3, (dec_ <= 3) | (lp > 3.5),
         lambda i: f"low GPS precision ({int(dec_[i])} decimals)" if dec_[i] <= 3 else "surveyor's GPS coordinates are unusually coarse")
    vl = F["village_loss_z"].to_numpy()
    vmed = loss.groupby(area).transform("median").to_numpy()
    rule(np.abs(vl) / 8, np.abs(vl) > 3.5,
         lambda i: f"loss {loss.iat[i]:.0f}% vs village median {vmed[i]:.0f}%")
    vd = F["surv_village_dev_z"].to_numpy()
    rule(np.abs(vd) / 8, np.abs(vd) > 3.5,
         lambda i: "surveyor's losses are systematically " + ("higher" if vd[i] > 0 else "lower") + " than village peers")
    dm = F["damage_mismatch"].fillna(0).to_numpy()
    rule(dm / 50, dm > C.TOTAL_DAMAGE_TOLERANCE_PCT, lambda i: f"total damage differs from affected x loss by {dm[i]:.0f} pts")
    a1, a2 = F["loss_zero_area_pos"].to_numpy(), F["loss_pos_area_zero"].to_numpy()
    rule(np.full(n, 0.4), (a1 | a2) > 0, lambda i: "affected area and crop loss contradict each other")
    fz = F["crop_farm_area_z"].to_numpy()
    rule(np.abs(fz) / 10, np.abs(fz) > 4,
         lambda i: f"farm area {farm.iat[i]:.3g} unusual for this crop")
    return F, rules


# ----------------------------------------------------------------------------------------------- A) risk model
def add_risk(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Return a copy of df with Risk_Score (0-100, relative priority) and Risk_Reasons (top 3, '; '-joined)."""
    from sklearn.ensemble import IsolationForest
    out = df.copy()
    if len(df) == 0:
        out[RISK_OUT[0]], out[RISK_OUT[1]] = pd.Series(dtype=float), pd.Series(dtype=object)
        return out
    F, rules = compute_features(df)
    X = F.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)
    Xr = X.rank(pct=True, method="average")          # rank transform: robust to heavy tails / zero inflation
    nzv = Xr.columns[Xr.nunique() > 1]
    Xr = Xr[nzv]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = IsolationForest(n_estimators=150, max_samples=min(len(Xr), 4096), random_state=seed, n_jobs=1).fit(Xr)
    s_iso = pd.Series(-iso.score_samples(Xr), index=df.index).rank(pct=True)
    zcols = [c for c in F.columns if c.endswith("_z") and c != "surveyor_day_z"] + ["surveyor_day_z"]
    Z = F[zcols].abs().to_numpy()
    top3 = -np.sort(-Z, axis=1)[:, :3].mean(axis=1)
    s_rz = pd.Series(top3, index=df.index).rank(pct=True)
    rule_max = np.max([r[0] for r in rules], axis=0) if rules else np.zeros(len(df))
    rule_sum = np.sum([r[0] for r in rules], axis=0) if rules else np.zeros(len(df))
    s_rule = pd.Series(rule_sum, index=df.index).rank(pct=True).where(rule_sum > 0, 0.0)
    blend = 0.4 * s_iso + 0.3 * s_rz + 0.3 * s_rule
    score = blend.rank(pct=True) * 100
    # rows with a strong explicit rule hit never score below 60 (so the list is not purely relative)
    score = np.where(rule_max >= 0.6, np.maximum(score, 60), score)
    W = np.column_stack([r[0] for r in rules]) if rules else np.zeros((len(df), 0))
    T = np.column_stack([r[1] for r in rules]) if rules else np.empty((len(df), 0), object)
    order = np.argsort(-W, axis=1, kind="stable")[:, :3]
    reasons = []
    for i in range(len(df)):
        picked = [T[i, j] for j in order[i] if W[i, j] > 0]
        reasons.append("; ".join(picked))
    out["Risk_Score"] = np.round(score, 1)
    out["Risk_Reasons"] = reasons
    return out


# ----------------------------------------------------------------------------------------------- B) supervised
CAT_COLS = ["crop_name", "calamity_type", "district", "tehsil"]


def ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for the classifiers: numeric features + a few categoricals + optional AI-output signals.
    Never uses the human QC label columns."""
    F, _ = compute_features(df)
    for c in CAT_COLS:
        F[c] = df[c].fillna("NA").astype(str).str.strip().str.lower() if c in df else "NA"
    conf = df["AI_Confidence"].map({"Low": 0, "Medium": 1, "High": 2}) if "AI_Confidence" in df else np.nan
    F["ai_confidence"] = conf
    F["ai_flag_count"] = df["AI_Flags"].fillna("").astype(str).map(lambda s: len([x for x in s.split(",") if x.strip()])) if "AI_Flags" in df else np.nan
    F["data_flag_count"] = df["Data_QC_Flags"].fillna("").astype(str).map(lambda s: len([x for x in s.split(",") if x.strip()])) if "Data_QC_Flags" in df else np.nan
    return F.replace([np.inf, -np.inf], np.nan)


def _clean_label(s: pd.Series) -> pd.Series:
    s = s.astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
    s = s.where(s.notna() & (s.astype(str) != "") & (s.astype(str).str.len() <= 40), None)  # long text = header/description row
    return s.map(lambda v: v.lower() if isinstance(v, str) else v)


def _single_thread():
    """Small tabular data: single-threaded OpenMP is faster (no oversubscription) and reproducible."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def _pipeline(kind: str, num_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10, max_categories=30)
    if kind == "logreg":
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
            ("cat", ohe, cat_cols)])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    else:
        pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", ohe, cat_cols)], sparse_threshold=0.0)
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, class_weight="balanced", random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def _cv_eval(kind, X, y, groups, num_cols, cat_cols, folds=5):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
    min_class = int(y.value_counts().min())
    k = max(2, min(folds, min_class))
    if groups is not None and groups.nunique() >= 2 * k:
        cv, kw, cvtype = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED), {"groups": groups}, "grouped by surveyor"
    else:
        cv, kw, cvtype = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED), {}, "stratified"
    with _single_thread():
        pred = cross_val_predict(_pipeline(kind, num_cols, cat_cols), X, y, cv=cv, **kw)
    return {"model": kind, "cv": f"{k}-fold {cvtype}", "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def _load(path: str) -> pd.DataFrame:
    import ingestion
    return ingestion.load(path)


def _needs_correction(df: pd.DataFrame, ai_path: str) -> pd.Series:
    """1 if the human-reviewed row differs from the AI output in any comparable QC field, else 0 (NaN if unmatched)."""
    ai = _load(ai_path)
    key = "docket_id"
    ai = ai.drop_duplicates(key).set_index(key)
    res = pd.Series(np.nan, index=df.index)
    m = df[key].isin(ai.index)
    diff = pd.Series(0, index=df.index)
    for c in COMPARE_COLS:
        if c in df and c in ai:
            a = _clean_label(df.loc[m, c])
            b = _clean_label(df.loc[m, key].map(ai[c]))
            both = a.notna() & b.notna()
            diff.loc[m] += (both & (a.astype(str) != b.astype(str))).astype(int)
    res[m] = (diff[m] > 0).astype(int)
    return res


def train(input_path, model_path, ai_output=None, min_labelled=MIN_LABELLED):
    import joblib
    df = _load(input_path)
    targets = {}
    for col, short in TARGETS.items():
        if col in df:
            targets[short] = _clean_label(df[col])
    if ai_output:
        nc = _needs_correction(df, ai_output)
        targets[NEEDS_CORRECTION] = nc.map({1: "yes", 0: "no"})
    X = ml_features(df)
    cat_cols = [c for c in CAT_COLS if c in X]
    num_cols = [c for c in X.columns if c not in cat_cols and X[c].notna().any()]  # all-NaN columns break HGB
    groups = _surveyor_key(df)
    bundle = {"version": 1, "num_cols": num_cols, "cat_cols": cat_cols, "targets": {}}
    report, skipped = [], []
    for short, y in targets.items():
        lab = y.notna()
        nlab = int(lab.sum())
        if nlab < min_labelled:
            skipped.append(f"{short}: only {nlab} labelled rows (need >= {min_labelled})")
            continue
        yy = y[lab].astype(str)
        vc = yy.value_counts()
        keep = vc[vc >= 5].index
        yy = yy[yy.isin(keep)]
        if len(yy) < min_labelled or yy.nunique() < 2:
            skipped.append(f"{short}: fewer than {min_labelled} usable rows or only one class after removing classes with <5 examples")
            continue
        Xl, gl = X.loc[yy.index], groups.loc[yy.index]
        base = float(yy.value_counts(normalize=True).iloc[0])
        results = [_cv_eval(k, Xl, yy, gl, num_cols, cat_cols) for k in ("gboost", "logreg")]
        best = max(results, key=lambda r: r["macro_f1"])
        with _single_thread():
            final = _pipeline(best["model"], num_cols, cat_cols).fit(Xl, yy)
        bundle["targets"][short] = {"pipeline": final, "classes": list(final.classes_), "n": int(len(yy)),
                                    "majority_baseline_accuracy": base, "cv_results": results, "chosen": best["model"]}
        report.append((short, len(yy), base, results, best["model"]))
    print(f"Training file: {input_path} ({len(df):,} rows)")
    for short, n, base, results, chosen in report:
        print(f"\n[{short}] n={n:,}  majority-class baseline accuracy={base:.3f}")
        for r in results:
            print(f"   {r['model']:7s} {r['cv']}: acc={r['accuracy']:.3f} bal_acc={r['balanced_accuracy']:.3f} macro_F1={r['macro_f1']:.3f}"
                  + ("  <- chosen" if r["model"] == chosen else ""))
        if max(r["macro_f1"] for r in results) < 0.5 or results[0]["accuracy"] <= base and results[1]["accuracy"] <= base:
            print("   note: no better than guessing the majority class; the tabular features carry little signal for this label.")
    for line in skipped:
        print(f"\nSKIPPED {line}")
    if not bundle["targets"]:
        print("\nNo model trained. This file has too few human-filled QC labels. Have reviewers fill the QC columns "
              f"(>= {min_labelled} rows per label), then re-run. Until then use the unsupervised Risk_Score "
              "(python learn_qc.py risk --input ...).")
        return 2
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"\nSaved model with targets {list(bundle['targets'])} -> {model_path}")
    print("Reminder: this learns reviewer patterns from tabular data; it does not improve the OCR/photo reading.")
    return 0


def add_predictions(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Add ML_<label> and ML_<label>_Conf columns (predicted class and its probability)."""
    import joblib
    b = joblib.load(model_path)
    X = ml_features(df).reindex(columns=b["num_cols"] + b["cat_cols"])
    out = df.copy()
    for short, t in b["targets"].items():
        with _single_thread():
            proba = t["pipeline"].predict_proba(X)
        out[f"ML_{short}"] = np.asarray(t["classes"], dtype=object)[proba.argmax(1)]
        out[f"ML_{short}_Conf"] = np.round(proba.max(1), 3)
    return out


def feedback(ai_path, human_path, out_path=None):
    """Per-field agreement between AI output and human-corrected output (joined on docket_id). Aggregates only."""
    ai, hu = _load(ai_path), _load(human_path)
    key = "docket_id"
    ai = ai.drop_duplicates(key).set_index(key)
    hu = hu.drop_duplicates(key).set_index(key)
    common = ai.index.intersection(hu.index)
    rows, conf = [], []
    for c in COMPARE_COLS:
        if c not in ai or c not in hu:
            continue
        if c in NUMERIC_COMPARE:
            a = pd.to_numeric(ai.loc[common, c], errors="coerce")
            h = pd.to_numeric(hu.loc[common, c], errors="coerce")
            ok = a.notna() & h.notna()
            agree = ((a - h).abs() <= NUMERIC_COMPARE[c])[ok]
            exact = ((a - h).abs() < 1e-9)[ok]
            n = int(ok.sum())
            rows.append({"field": c, "compared": n, "agreement_%": round(100 * agree.mean(), 1) if n else None,
                         "exact_%": round(100 * exact.mean(), 1) if n else None,
                         "note": f"numeric, tolerance +-{NUMERIC_COMPARE[c]} pts"})
        else:
            a, h = _clean_label(ai.loc[common, c]), _clean_label(hu.loc[common, c])
            ok = a.notna() & h.notna()
            n = int(ok.sum())
            agree = (a[ok].astype(str) == h[ok].astype(str))
            rows.append({"field": c, "compared": n, "agreement_%": round(100 * agree.mean(), 1) if n else None,
                         "exact_%": None, "note": "categorical"})
            bad = pd.DataFrame({"ai": a[ok][~agree], "human": h[ok][~agree]})
            for (x, y), cnt in bad.value_counts().head(5).items():
                conf.append({"field": c, "ai_said": x, "human_said": y, "count": int(cnt)})
    res = pd.DataFrame(rows)
    cf = pd.DataFrame(conf, columns=["field", "ai_said", "human_said", "count"])
    print(f"Matched {len(common):,} dockets (AI rows {len(ai):,}, human rows {len(hu):,})")
    if res.empty or res["compared"].sum() == 0:
        print("No fields with values in both files - nothing to compare (are the human QC columns filled?).")
    else:
        print(res.to_string(index=False))
        weak = res[(res["compared"] >= 30) & (res["agreement_%"] < 90)]
        for _, r in weak.iterrows():
            print(f"  tune prompt: '{r['field']}' agrees only {r['agreement_%']}% - review the top confusions below")
        if len(cf):
            print(cf.to_string(index=False))
    if out_path:
        with pd.ExcelWriter(out_path) as w:
            res.to_excel(w, "Per_Field", index=False)
            cf.to_excel(w, "Top_Confusions", index=False)
        print(f"Wrote {out_path}")
    return res, cf


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description="CLAP QC learning layer (risk scoring + reviewer-pattern models)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("risk", help="unsupervised Risk_Score/Risk_Reasons; works without labels")
    r.add_argument("--input", required=True)
    r.add_argument("--output")
    t = sub.add_parser("train", help="train reviewer-pattern classifiers from a human-QC'd file")
    t.add_argument("--input", required=True)
    t.add_argument("--model", default="output/model.joblib")
    t.add_argument("--ai-output", help="AI-generated QC file for the same rows; enables the Needs_Correction target")
    t.add_argument("--min-labelled", type=int, default=MIN_LABELLED)
    pr = sub.add_parser("predict", help="add ML_<label> and confidence columns")
    pr.add_argument("--input", required=True)
    pr.add_argument("--model", required=True)
    pr.add_argument("--output")
    f = sub.add_parser("feedback", help="per-field AI vs human accuracy, to tune the OCR/photo prompts")
    f.add_argument("--ai", required=True)
    f.add_argument("--human", required=True)
    f.add_argument("--output")
    a = p.parse_args(argv)
    if a.cmd == "risk":
        out = add_risk(_load(a.input))
        dest = a.output or str(Path(a.input).with_suffix("")) + "_risk.xlsx"
        out.to_excel(dest, index=False)
        print(f"Wrote {dest}")
        return 0
    if a.cmd == "train":
        return train(a.input, a.model, a.ai_output, a.min_labelled)
    if a.cmd == "predict":
        out = add_predictions(_load(a.input), a.model)
        dest = a.output or str(Path(a.input).with_suffix("")) + "_ML.xlsx"
        out.to_excel(dest, index=False)
        print(f"Wrote {dest}")
        return 0
    feedback(a.ai, a.human, a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
