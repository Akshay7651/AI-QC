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
    blank = pd.Series("Unknown", index=df.index)
    dist = df["district"].fillna("Unknown") if "district" in df else blank
    surv_name = df["surveyor_name"].fillna("Unknown") if "surveyor_name" in df else blank
    t = pd.DataFrame({"District": dist, "QC Done": done, "Match": m.eq("Match"),
                      "Mismatch": m.eq("Mismatch"), "Flagged": flagged})
    t["Total"] = 1
    d = t.groupby("District").sum().reset_index()
    d["Match%"] = (100 * d["Match"] / (d["Match"] + d["Mismatch"]).replace(0, float("nan"))).round(1)
    district = d[["District", "Total", "QC Done", "Match", "Mismatch", "Flagged", "Match%"]]

    s = pd.DataFrame({"Surveyor": surv_name,
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


# ============================================================================ the big QC workbook (fast, atomic)
import os
import time
from datetime import datetime
from pathlib import Path

VERDICT_COLOURS = {"Reject-evidence": "#F8BBD0", "Manual-check": "#FFE0B2", "Review": "#FFF9C4", "OK": "#C8E6C9"}
CONF_COLOURS = {"Low": "#F8BBD0", "Medium": "#FFF9C4", "High": "#C8E6C9"}
BAD = "#F8BBD0"


from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as _ILLEGAL


def _cell_value(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d %H:%M:%S") if v == v else None
    if isinstance(v, str):
        return _ILLEGAL.sub("", v) if _ILLEGAL.search(v) else v
    if isinstance(v, (bool, int, float)):
        return v
    if hasattr(v, "item"):
        try:
            return _cell_value(v.item())
        except Exception:
            pass
    return str(v)


def write_xlsx(df: pd.DataFrame, path: str, remarks_col="Any Other Remarks"):
    """One sheet, frozen header + docket column, widths, wrapped remarks, colour rules. xlsxwriter (constant memory) if present,
    else openpyxl write-only. Returns the writer name."""
    cols = [str(c) for c in df.columns]
    try:
        import xlsxwriter
    except ImportError:
        return _write_openpyxl(df, path, cols)
    wb = xlsxwriter.Workbook(path, {"constant_memory": True, "nan_inf_to_errors": True, "strings_to_formulas": False,
                                    "strings_to_urls": False})
    ws = wb.add_worksheet("QC")
    head = wb.add_format({"bold": True, "bg_color": "#1F3A5F", "font_color": "#FFFFFF", "text_wrap": True, "valign": "top", "border": 1})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    sample = df.head(300)
    widths = []
    for c in cols:
        s = sample[c].astype(str).str.len() if len(sample) else pd.Series([0])
        w = min(60, max(len(c) * 0.9, float(s.quantile(0.9)) if len(s) else 8) + 2)
        widths.append(max(8, w))
    wrap_cols = {cols.index(remarks_col)} if remarks_col in cols else set()
    for j, c in enumerate(cols):
        if j in wrap_cols:
            ws.set_column(j, j, 110, wrap)
        else:
            ws.set_column(j, j, widths[j])
    ws.set_row(0, 45)
    ws.write_row(0, 0, cols, head)
    arr = df.astype(object).to_numpy()
    for r in range(arr.shape[0]):
        row = arr[r]
        for j in range(len(cols)):
            v = _cell_value(row[j])
            if v is None:
                continue
            if isinstance(v, str):
                ws.write_string(r + 1, j, v)
            elif isinstance(v, bool):
                ws.write_boolean(r + 1, j, v)
            else:
                ws.write_number(r + 1, j, v)
    n = len(df) + 1
    ws.freeze_panes(1, 1 if cols and cols[0].lower().startswith(("docket", "application")) else 0)
    if len(cols):
        ws.autofilter(0, 0, max(n - 1, 1), len(cols) - 1)

    def cf(col, criteria, colour):
        if col in cols:
            j = cols.index(col)
            ws.conditional_format(1, j, max(n, 2), j, {"type": "cell", "criteria": "==", "value": f'"{criteria}"',
                                                        "format": wb.add_format({"bg_color": colour})})
    for k, colour in VERDICT_COLOURS.items():
        cf("QC Verdict", k, colour)
    for k, colour in CONF_COLOURS.items():
        cf("AI_Confidence", k, colour)
    for c in cols:
        if c.startswith("Match/Mismatch") or c.startswith("Form vs App"):
            cf(c, "Mismatch", BAD)
            cf(c, "Match", "#C8E6C9")
        elif "Signature (Yes/No)" in c:
            cf(c, "No", BAD)
        elif c.startswith("Photo is form image"):
            cf(c, "Yes", BAD)
        elif c.startswith("PO ID matches"):
            cf(c, "No", BAD)
        elif c.startswith("Form Status"):
            cf(c, "overwrite", BAD)
            cf(c, "incomplete", "#FFF9C4")
    wb.close()
    return "xlsxwriter"


def _write_openpyxl(df, path, cols):
    from openpyxl import Workbook
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    wb = Workbook(write_only=True)
    ws = wb.create_sheet("QC")
    ws.freeze_panes = "A2"
    ws.append(cols)
    for row in df.astype(object).to_numpy():
        out = []
        for v in row:
            v = _cell_value(v)
            out.append(ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)
        ws.append(out)
    wb.save(path)
    return "openpyxl"


class ExcelTarget:
    """Atomic writes (tmp + replace). If the file is locked (Excel open on Windows) fall back to a timestamped sibling."""

    def __init__(self, path):
        self.path = str(path)
        self.fallback = None
        self.last_note = ""

    def write(self, df):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.path}.tmp{os.getpid()}.xlsx"
        try:
            write_xlsx(df, tmp)
            try:
                os.replace(tmp, self.path)
                self.last_note = ""
                return self.path
            except PermissionError:
                fb = self.fallback or str(Path(self.path).with_name(
                    f"{Path(self.path).stem}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"))
                os.replace(tmp, fb)
                self.fallback = fb
                self.last_note = f"{self.path} is open/locked - saved to {fb} instead"
                return fb
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
