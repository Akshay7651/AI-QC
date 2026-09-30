"""Summary workbook: District Summary, Surveyor Report, Issue Summary."""
import pandas as pd

import config as C


def _truthy(s):
    return s.astype(str).str.upper().eq("TRUE")


def build(df: pd.DataFrame, path: str):
    flagged = df["Suggested_Remark"].fillna("OK").ne("OK") if "Suggested_Remark" in df else pd.Series(False, index=df.index)
    if "Data_QC_Flags" in df:
        flagged |= df["Data_QC_Flags"].fillna("").ne("")
    m = df[C.COL_MATCH] if C.COL_MATCH in df else pd.Series("NA", index=df.index)
    done = _truthy(df[C.COL_QC_DONE]) if C.COL_QC_DONE in df else pd.Series(False, index=df.index)
    t = pd.DataFrame({"District": df["district"].fillna("Unknown"), "QC Done": done, "Match": m.eq("Match"),
                      "Mismatch": m.eq("Mismatch"), "Flagged": flagged})
    t["Total"] = 1
    d = t.groupby("District").sum().reset_index()
    d["Match%"] = (100 * d["Match"] / (d["Match"] + d["Mismatch"]).replace(0, float("nan"))).round(1)
    district = d[["District", "Total", "QC Done", "Match", "Mismatch", "Flagged", "Match%"]]

    s = pd.DataFrame({"Surveyor": df["surveyor_name"].fillna("Unknown"),
                      "Same_Location_Flags": df.get("Suggested_Remark", pd.Series("", index=df.index)).fillna("").str.startswith("Same Location"),
                      "Mismatch_Count": m.eq("Mismatch")})
    s["Records"] = 1
    sv = s.groupby("Surveyor").sum().reset_index()
    sv["Consistency_Score"] = (100 - 100 * (sv["Same_Location_Flags"] + sv["Mismatch_Count"]) / sv["Records"]).clip(0, 100).round(1)
    surveyor = sv[["Surveyor", "Records", "Same_Location_Flags", "Mismatch_Count", "Consistency_Score"]]

    issues = []
    for col in ("Data_QC_Flags", "AI_Flags"):
        if col in df:
            for v in df[col].fillna(""):
                issues += [x.strip() for x in v.split(",") if x.strip()]
    if "Suggested_Remark" in df:
        issues += [r for r in df["Suggested_Remark"].fillna("OK") if r != "OK"]
    iss = pd.Series(issues, dtype=str).value_counts().rename_axis("Issue_Type").reset_index(name="Count")
    iss["% of Total"] = (100 * iss["Count"] / max(len(df), 1)).round(2)

    with pd.ExcelWriter(path) as w:
        district.to_excel(w, sheet_name="District Summary", index=False)
        surveyor.to_excel(w, sheet_name="Surveyor Report", index=False)
        iss.to_excel(w, sheet_name="Issue Summary", index=False)
