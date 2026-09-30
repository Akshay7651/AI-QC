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


def write_xlsx_slow(df: pd.DataFrame, path: str, remarks_col="Any Other Remarks"):
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


# ---- fast direct-XML writer (about 10x faster than xlsxwriter for 40k x 80 cells) -------------------------------
import re as _re
import zipfile as _zip
from xml.sax.saxutils import escape as _esc

_CTRL = _re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def _col_letters(n):
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


_DXF = {"red": "F8BBD0", "amber": "FFE0B2", "yellow": "FFF9C4", "green": "C8E6C9"}


def write_xlsx(df: pd.DataFrame, path: str, remarks_col="Any Other Remarks"):
    """One sheet 'QC': frozen header+first id column, widths, wrapped remarks, colour rules, autofilter. Streams XML straight
    into the zip (no per-cell library overhead). Falls back to xlsxwriter/openpyxl if anything goes wrong."""
    try:
        return _write_fast(df, path, remarks_col)
    except Exception:
        return write_xlsx_slow(df, path, remarks_col)


def _write_fast(df, path, remarks_col):
    cols = [str(c) for c in df.columns]
    nc, nr = len(cols), len(df)
    L = [_col_letters(j) for j in range(nc)]
    sample = df.head(300)
    widths = []
    for c in cols:
        s = sample[c].astype(str).str.len() if nr else pd.Series([0])
        widths.append(max(8, min(60, max(len(c) * 0.9, float(s.quantile(0.9)) if len(s) else 8) + 2)))
    rem_j = cols.index(remarks_col) if remarks_col in cols else -1
    colxml = "".join(f'<col min="{j + 1}" max="{j + 1}" width="{110 if j == rem_j else widths[j]:.1f}" customWidth="1"'
                     + (' style="2"' if j == rem_j else "") + "/>" for j in range(nc))
    # conditional formats: (column index, value, dxf id)
    dxf_ids = {k: i for i, k in enumerate(_DXF)}
    rules = []

    def cf(c, val, colour):
        if c in cols:
            rules.append((cols.index(c), val, dxf_ids[colour]))
    for k, colour in {"Reject-evidence": "red", "Manual-check": "amber", "Review": "yellow", "OK": "green"}.items():
        cf("QC Verdict", k, colour)
    for k, colour in {"Low": "red", "Medium": "yellow", "High": "green"}.items():
        cf("AI_Confidence", k, colour)
    for c in cols:
        if c.startswith("Match/Mismatch") or c.startswith("Form vs App"):
            cf(c, "Mismatch", "red")
            cf(c, "Match", "green")
        elif "Signature (Yes/No)" in c or c.startswith("PO ID matches"):
            cf(c, "No", "red")
        elif c.startswith("Photo is form image"):
            cf(c, "Yes", "red")
        elif c.startswith("Form Status"):
            cf(c, "overwrite", "red")
            cf(c, "incomplete", "yellow")
    last = nr + 1
    cfxml, prio = [], 1
    for j, val, d in rules:
        cfxml.append(f'<conditionalFormatting sqref="{L[j]}2:{L[j]}{max(last, 2)}"><cfRule type="cellIs" dxfId="{d}" priority="{prio}" '
                     f'operator="equal"><formula>"{_esc(val)}"</formula></cfRule></conditionalFormatting>')
        prio += 1
    xsplit = 1 if cols and cols[0].lower().startswith(("docket", "application")) else 0
    pane = (f'<pane xSplit="{xsplit}" ySplit="1" topLeftCell="{"B" if xsplit else "A"}2" activePane="{"bottomRight" if xsplit else "bottomLeft"}" state="frozen"/>')
    head = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<sheetViews><sheetView workbookViewId="0">{pane}</sheetView></sheetViews><sheetFormatPr defaultRowHeight="15"/>'
            f'<cols>{colxml}</cols><sheetData>')
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
              '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts>'
              '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
              '<fill><patternFill patternType="solid"><fgColor rgb="FF1F3A5F"/><bgColor indexed="64"/></patternFill></fill></fills>'
              '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
              '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
              '<cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
              '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf>'
              '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1"><alignment vertical="top" wrapText="1"/></xf></cellXfs>'
              '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
              f'<dxfs count="{len(_DXF)}">' + "".join(f'<dxf><fill><patternFill patternType="solid"><fgColor rgb="FF{v}"/><bgColor rgb="FF{v}"/></patternFill></fill></dxf>'
                                                    for v in _DXF.values()) + '</dxfs></styleSheet>')
    wbxml = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
             'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="QC" sheetId="1" r:id="rId1"/></sheets></workbook>')
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
    wbrels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
              '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
              '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>')
    ctypes = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
              '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
              '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
              '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
              '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>')
    esc, ctrl = _esc, _CTRL

    def rows():
        yield head
        yield '<row r="1" ht="45" customHeight="1">' + "".join(
            f'<c r="{L[j]}1" s="1" t="inlineStr"><is><t>{esc(c)}</t></is></c>' for j, c in enumerate(cols)) + "</row>"
        arr = df.astype(object).to_numpy()
        buf = []
        for r in range(nr):
            rn = r + 2
            cells = []
            row = arr[r]
            for j in range(nc):
                v = row[j]
                if v is None:
                    continue
                t = type(v)
                if t is str:
                    if not v:
                        continue
                    if ctrl.search(v):
                        v = ctrl.sub("", v)
                    if len(v) > 32000:
                        v = v[:32000]
                    cells.append(f'<c r="{L[j]}{rn}" t="inlineStr"><is><t xml:space="preserve">{esc(v)}</t></is></c>')
                elif t is bool:
                    cells.append(f'<c r="{L[j]}{rn}" t="b"><v>{int(v)}</v></c>')
                elif t is int or t is float:
                    if v != v or v in (float("inf"), float("-inf")):
                        continue
                    cells.append(f'<c r="{L[j]}{rn}"><v>{v!r}</v></c>')
                else:
                    v = _cell_value(v)
                    if v is None:
                        continue
                    if isinstance(v, bool):
                        cells.append(f'<c r="{L[j]}{rn}" t="b"><v>{int(v)}</v></c>')
                    elif isinstance(v, (int, float)):
                        cells.append(f'<c r="{L[j]}{rn}"><v>{v!r}</v></c>')
                    else:
                        cells.append(f'<c r="{L[j]}{rn}" t="inlineStr"><is><t xml:space="preserve">{esc(str(v))}</t></is></c>')
            buf.append(f'<row r="{rn}">' + "".join(cells) + "</row>")
            if len(buf) >= 2000:
                yield "".join(buf)
                buf = []
        if buf:
            yield "".join(buf)
        yield ("</sheetData>" + (f'<autoFilter ref="A1:{L[-1]}{max(last, 2)}"/>' if nc else "") + "".join(cfxml) + "</worksheet>")

    with _zip.ZipFile(path, "w", _zip.ZIP_DEFLATED, compresslevel=3) as z:
        z.writestr("[Content_Types].xml", ctypes)
        z.writestr("_rels/.rels", rels)
        z.writestr("xl/workbook.xml", wbxml)
        z.writestr("xl/_rels/workbook.xml.rels", wbrels)
        z.writestr("xl/styles.xml", styles)
        with z.open("xl/worksheets/sheet1.xml", "w", force_zip64=True) as f:
            for chunk in rows():
                f.write(chunk.encode("utf-8"))
    return "fastxml"
