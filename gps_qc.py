"""GPS proximity engine: KDTree neighbours, connected-component clustering, scoring. No API calls."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import config as C

OUT = ["Nearby_Same_Surveyor_25m", "Nearby_Any_Surveyor_25m", "Records_On_Same_Field",
       "Group_ID", "Cluster_Size", "GPS_Score", "Suggested_Remark"]


def run(df: pd.DataFrame, radius_m: float = C.GPS_PROXIMITY_RADIUS_M) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame(index=df.index, columns=OUT)
    valid = (df["latitude"].notna() & df["longitude"].notna()).to_numpy()
    same_surv = np.zeros(n, int)
    any_surv = np.zeros(n, int)
    group = np.full(n, -1)
    csize = np.ones(n, int)

    vi = np.flatnonzero(valid)
    if len(vi):
        lat = df["latitude"].to_numpy(float)[vi]
        lng = df["longitude"].to_numpy(float)[vi]
        # local equirectangular projection to metres
        xy = np.column_stack([lat * 111_320.0, lng * 111_320.0 * np.cos(np.radians(lat.mean()))])
        pairs = cKDTree(xy).query_pairs(radius_m, output_type="ndarray")
        surv = pd.factorize(df["surveyor_name"].fillna("").astype(str).str.strip().str.lower())[0][vi]
        m = len(vi)
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            any_v = np.bincount(a, minlength=m) + np.bincount(b, minlength=m)
            same = surv[a] == surv[b]
            same_v = np.bincount(a[same], minlength=m) + np.bincount(b[same], minlength=m)
            g = coo_matrix((np.ones(len(a)), (a, b)), shape=(m, m))
        else:
            any_v = same_v = np.zeros(m, int)
            g = coo_matrix((m, m))
        _, labels = connected_components(g, directed=False)
        sizes = np.bincount(labels)
        any_surv[vi], same_surv[vi] = any_v, same_v
        group[vi] = labels + 1
        csize[vi] = sizes[labels]
    # invalid-GPS rows get their own unique group ids after the valid ones
    inv = np.flatnonzero(~valid)
    group[inv] = (group.max() if n else 0) + 1 + np.arange(len(inv))

    field_key = df[["khasra_number", "division_number", "village"]].astype(str).agg("|".join, axis=1)
    on_field = field_key.groupby(field_key).transform("size").to_numpy()
    on_field = np.where(df["khasra_number"].isna(), 1, on_field)

    loss = df["crop_loss_pct"].to_numpy(float)
    score = np.zeros(n, int)
    score += 40 * (same_surv > 5)
    score += 20 * (same_surv > 10)
    score += 15 * ((loss < 15) & (same_surv > 3))
    score += 10 * (on_field > C.SAME_FIELD_FLAG_COUNT)
    score += 15 * (csize > 20)
    remark = np.select([score >= 75, score >= 55, score >= 35],
                       ["Same Location - QC Required", "Same Location - Low Damage", "Review - Multiple Records"], "OK")
    out["Nearby_Same_Surveyor_25m"] = same_surv
    out["Nearby_Any_Surveyor_25m"] = any_surv
    out["Records_On_Same_Field"] = on_field
    out["Group_ID"] = group
    out["Cluster_Size"] = csize
    out["GPS_Score"] = score
    out["Suggested_Remark"] = remark
    return out
