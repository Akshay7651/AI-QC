"""Rajasthan (National Insurance, Jaipur Regional Office) crop-loss survey form: a different template from the Haryana Proforma-3.

No barcode / form number, no total row.  Lower table: 9 columns (no., application id, docket id, crop, khasra, khata, insured area,
affected area %, loss %), 10 rows; one form can list several dockets.  Signatures: farmer | insurance company TC/DC | agriculture supervisor (AAO).
Reading: the whole-cell CNN (digits.read_cell) on the last two columns of every row.
"""
import re

import cv2
import numpy as np

import digits as D
import form_p3 as P


def is_rajasthan(rgb):
    """True when the printed header says National Insurance / Jaipur Regional Office (Tesseract on the top of the page)."""
    import pytesseract
    h = rgb.shape[0]
    top = cv2.cvtColor(rgb[: int(0.2 * h)], cv2.COLOR_RGB2GRAY)
    top = cv2.resize(top, None, fx=1.0, fy=1.0)
    try:
        t = pytesseract.image_to_string(top, lang="eng", config="--psm 6").lower()
    except Exception:     # noqa: BLE001
        return False
    return bool(re.search(r"national\s*insur|regional\s*office|jaipur", t))


def rows_geometry(lay):
    """(y0, p): top of row 1 and row pitch in 1000-wide geometry coords.  (Proforma-3's header is 1.93 pitches tall, this one about 2.4,
    so form_p3.locate_table lands two rows too low: take the first line of the row chain instead.)"""
    g = lay["g"]
    lines = P.hlines(lay["hl"], thr_frac=0.22)
    ch = P.fit_row_chain(lines, g.shape[1])
    if ch is None or ch["n_hit"] < 9:
        return None
    return float(ch["y_first"]), float(ch["p"])


def column_edges(lay, y0, p):
    """-> (edges, s): x (1000-wide geometry coords) of [table left, area-column left, loss-column left, table right] and the column pitch s.
    The right edge comes from the length of the ruled ROW lines (robust against handwriting confusing the vertical-line detector); the last
    two columns (affected area %, loss %) are equally wide, about 11.3% of the table; each edge is then snapped to a detected vertical line."""
    hl, vl, g = lay["hl"], lay["vl"], lay["g"]
    W = g.shape[1]
    ext = []
    for k in range(0, 11):
        e = P.line_extent(hl, y0 + k * p, 0.012 * W)
        if e is not None and e[1] - e[0] > 0.45 * W:
            ext.append(e)
    if len(ext) < 3:
        return None
    Lx = float(np.median([e[0] for e in ext]))
    Rx = float(np.median([e[1] for e in ext]))
    w = Rx - Lx
    if w < 0.45 * W:
        return None
    s = 0.113 * w
    ya, yb = int(y0 + 0.2 * p), int(y0 + 9.8 * p)
    vp = (vl[ya:yb] > 0).sum(0).astype(float)
    vp = vp + np.r_[vp[1:], 0] + np.r_[0, vp[:-1]]
    vx = np.array(sorted(x for x, _ in P._peaks1d(vp, 0.22 * (yb - ya), 3)))
    pred = [Rx - 2 * s, Rx - s, Rx]
    snapped, hit = [], []
    for q in pred:
        ok = bool(len(vx) and np.min(np.abs(vx - q)) < 0.03 * w)
        hit.append(ok)
        snapped.append(float(vx[np.argmin(np.abs(vx - q))]) if ok else float(q))
    # the ruled lines fade out on some photos, then the real right edge is unknown: a vertical border must exist at the right edge
    # (and one more column edge, unless the table fills the photo); otherwise decline rather than read the wrong cells
    if not (hit[2] and (sum(hit) >= 2 or w >= 0.8 * W)) and not (len(ext) >= 9 and w >= 0.5 * W):
        return None
    a0, a1, l1 = snapped
    return [Lx, a0, a1, l1], float(np.mean([a1 - a0, l1 - a1]))


def read_rows(lay, read_cell):
    r, sc = lay["rgb"], lay["scale"]
    geo = rows_geometry(lay)
    if geo is None:
        return None
    y0, p = geo
    ce = column_edges(lay, y0, p)
    if ce is None:
        return None
    edges, s = ce
    a0, a1, l1 = edges[1], edges[2], edges[3]
    rows = []
    for k in range(10):
        ya, yb = y0 + k * p, y0 + (k + 1) * p
        pad = 0.08 * (yb - ya)
        out = {}
        for nm, (x0, x1) in (("area", (a0, a1)), ("loss", (a1, l1))):
            box = tuple(int(round(v * sc)) for v in (x0, ya - pad, x1, yb + pad))
            out[nm] = read_cell(P.crop(r, box, 2))
        rows.append(out)
    return rows, (a0, a1, l1), (y0, p)


def cell_crops(lay):
    """-> (list of 10 (area_crop, loss_crop) RGB arrays, geometry) or None."""
    r, sc = lay["rgb"], lay["scale"]
    geo = rows_geometry(lay)
    if geo is None:
        return None
    y0, p = geo
    ce = column_edges(lay, y0, p)
    if ce is None:
        return None
    edges, s = ce
    a0, a1, l1 = edges[1], edges[2], edges[3]
    out = []
    for k in range(10):
        ya, yb = y0 + k * p, y0 + (k + 1) * p
        pad = 0.08 * (yb - ya)
        pair = []
        for x0, x1 in ((a0, a1), (a1, l1)):
            box = tuple(int(round(v * sc)) for v in (x0, ya - pad, x1, yb + pad))
            pair.append(P.crop(r, box, 2))
        out.append(tuple(pair))
    return out, (a0, a1, l1, y0, p)


def ink_share(crop):
    img = D.cell_image(crop)
    return float((img > 127).mean())


def read_form_raj(lay, out, cell_gate, notes):
    """Fill the common read_form() output for a Rajasthan form.  Returns True when the table could be read."""
    res = read_rows(lay, D.read_cell)
    if res is None:
        notes.append("Rajasthan form: table rows could not be located")
        return False
    rows, _, _ = res
    out["template"] = "rajasthan"
    out["is_proforma3"] = True
    out["po_id"], out["po_id_matches"], out["form_no"], out["form_no_conf"] = None, None, None, 0.0
    out["total_row_blank"] = True                   # this template has no total row
    out["farmer_signed"] = out["company_signed"] = out["worker_signed"] = out["officer_signed"] = None     # not assessed unless a trained classifier exists
    try:
        sp = sig_pred(sig_feats(lay))
        out["farmer_signed"], out["company_signed"], out["officer_signed"] = sp["farmer"], sp["company"], sp["aao"]
    except Exception as e:     # noqa: BLE001
        notes.append(f"signature analysis failed: {type(e).__name__}")
    cells, filled = [], []
    for k, r in enumerate(rows):
        a, l = r["area"], r["loss"]
        empty = a["cls"] == "EMPTY" and l["cls"] == "EMPTY" and min(a["conf"], l["conf"]) >= 0.75
        cells.append({"row": k + 1, "area": a["value"], "loss": l["value"], "area_conf": round(a["conf"], 3), "loss_conf": round(l["conf"], 3),
                      "area_cls": a["cls"], "loss_cls": l["cls"], "empty": empty})
        if not empty:
            filled.append(k)
    out["raj_rows"] = cells
    fc = out["field_conf"]
    if len(filled) == 1:
        c = cells[filled[0]]
        ok = c["area_conf"] >= cell_gate and c["loss_conf"] >= cell_gate and c["area"] is not None and c["loss"] is not None
        if ok:
            out["row_area"], out["row_loss"] = c["area"], c["loss"]
            fc["row_area"], fc["row_loss"] = c["area_conf"], c["loss_conf"]
        else:
            notes.append("handwritten area/loss not confidently readable - verify manually")
    elif len(filled) > 1:
        notes.append(f"{len(filled)} rows are filled on this form (several dockets); the row of this docket was not singled out")
        out["raj_pairs"] = [(cells[k]["area"], cells[k]["loss"]) for k in filled
                            if cells[k]["area_conf"] >= cell_gate and cells[k]["loss_conf"] >= cell_gate and cells[k]["area"] is not None and cells[k]["loss"] is not None]
        out["raj_all_read"] = len(out["raj_pairs"]) == len(filled)
    else:
        notes.append("no filled row found in the table")
    return True


# ----------------------------------------------------------------------------------------- signatures
def sig_boxes(lay):
    """RGB crops of the three signature blocks: farmer | insurance company TC/DC | agriculture supervisor (AAO).
    They sit above the printed 'hastakshar' labels, about 4.2..7.6 row pitches under the table."""
    r, sc = lay["rgb"], lay["scale"]
    geo = rows_geometry(lay)
    ce = column_edges(lay, *geo) if geo else None
    if not geo or not ce:
        return None, None
    y0, p = geo
    edges = ce[0]
    L, R = edges[0], edges[-1]
    w = R - L
    ya, yb = (y0 + 10 * p + 4.2 * p) * sc, (y0 + 10 * p + 7.6 * p) * sc
    out = {}
    for nm, (a, b) in (("farmer", (0.0, 0.34)), ("company", (0.33, 0.67)), ("aao", (0.66, 1.0))):
        x0, x1 = (L + a * w) * sc, (L + b * w) * sc
        out[nm] = r[int(max(0, ya)): int(min(r.shape[0], yb)), int(max(0, x0)): int(min(r.shape[1], x1))]
    return out, p * sc


def sig_feats(lay):
    import form_reader as FR
    boxes, p = sig_boxes(lay)
    if boxes is None:
        return None
    return {nm: FR.sig_features(b, p, split=0.0) for nm, b in boxes.items()}


_SIG = None


def sig_pred(feats):
    """-> dict farmer/company/aao -> True/False/None from models/sig_clf_raj.joblib (None when not trained / not trusted)"""
    global _SIG
    import os
    if _SIG is None:
        try:
            import joblib
            _SIG = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "sig_clf_raj.joblib"))
        except Exception:     # noqa: BLE001
            _SIG = False
    out = {"farmer": None, "company": None, "aao": None}
    if not _SIG or not feats:
        return out
    for nm, m in _SIG["models"].items():
        if m is not None and feats.get(nm):
            p = float(m["model"].predict_proba([[feats[nm][k] for k in _SIG["keys"]]])[0][1])
            if m["t_hi"] is not None and p >= m["t_hi"]:
                out[nm] = True
            elif m["t_lo"] is not None and p <= m["t_lo"]:
                out[nm] = False
    return out
