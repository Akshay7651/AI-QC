"""GPS proximity engine: KDTree neighbours, connected-component clustering, scoring. No API calls."""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

import config as C

OUT = ["Nearby_Same_Surveyor_25m", "Nearby_Any_Surveyor_25m", "Records_On_Same_Field",
       "Group_ID", "Cluster_Size", "Suggested_Remark", "Suggest_%"]


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
        sname = df["surveyor_name"].fillna("").astype(str).str.strip().str.lower()
        codes = pd.factorize(sname)[0]
        blank = (sname == "").to_numpy()  # unknown surveyor identity is never treated as 'same surveyor'
        codes = np.where(blank, codes.max(initial=-1) + 1 + np.arange(n), codes)
        surv = codes[vi]
        m = len(vi)
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            any_v = np.bincount(a, minlength=m) + np.bincount(b, minlength=m)
            same = surv[a] == surv[b]
            same_v = np.bincount(a[same], minlength=m) + np.bincount(b[same], minlength=m)
            g = coo_matrix((np.ones(int(same.sum())), (a[same], b[same])), shape=(m, m))
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

    # Same-field key = patwar circle (else village/tehsil) + land survey number, as in the Level-1 dashboard.
    area = df["patwar_circle"].where(df["patwar_circle"].notna(), df["village"]).where(lambda x: x.notna(), df["tehsil"]).fillna("")
    field_key = area.astype(str) + "|" + df["khasra_number"].fillna("").astype(str)
    on_field = field_key.groupby(field_key).transform("size").to_numpy()
    on_field = np.where(df["khasra_number"].isna(), 1, on_field)

    # Damage used for the low-damage rule: total damage %, falling back to crop loss %.
    dmg = df["total_damage_pct"].where(df["total_damage_pct"].notna(), df["crop_loss_pct"]).to_numpy(float)
    heavy = same_surv > C.GPS_SAME_SURVEYOR_MIN
    remark = np.select(
        [heavy & (dmg > C.GPS_DAMAGE_MIN), heavy, on_field > C.SAME_FIELD_FLAG_COUNT],
        ["Same Location - QC Required", "Same Location - Low Damage", "Review - Multiple Records"], "OK")
    conf = np.select([remark == "Same Location - QC Required", remark == "Same Location - Low Damage",
                      remark == "Review - Multiple Records"], [95, 75, 65], 55)
    # Group_ID: only clusters of >1 same-surveyor points get an id (G-001...)
    gsize = csize
    first = {}
    gid = np.full(n, "", dtype=object)
    for i in range(n):
        if valid[i] and gsize[i] > 1:
            gid[i] = first.setdefault(group[i], f"G-{len(first) + 1:03d}")
    out["Nearby_Same_Surveyor_25m"] = same_surv
    out["Nearby_Any_Surveyor_25m"] = any_surv
    out["Records_On_Same_Field"] = on_field
    out["Group_ID"] = gid
    out["Cluster_Size"] = csize
    out["Suggested_Remark"] = remark
    out["Suggest_%"] = conf
    return out
