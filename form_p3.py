"""Proforma-3 (PMFBY crop-loss assessment report) geometry: page normalisation, table / grid / signature-box location.

Pipeline (all geometry is done on a 1000 px wide grey copy, crops are cut from a ~1500 px wide colour copy):
  load -> orientation (Hough on ruled lines gives skew + portrait/landscape; 0/180 decided by the printed BARCODE being
  in the top-right) -> deskew -> locate the lower table (equally spaced ruled lines, 9 vertical lines), the upper
  field grid (fields 1-15) and the signature label band -> `Layout` (pixel boxes in the hi-res normalised image).

Everything is expressed relative to the detected table (pitch = row height), so page position / zoom does not matter.
Reading of digits lives in digits.py, orchestration in form_reader.py.
"""
import io
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageOps

GW = 1000          # width of the geometry image
HW = 1500          # target width of the hi-res image used for crops

# Column edges of the lower table as a fraction of (right-left) measured on a reference form
COL_FR = np.array([0.0, 0.051, 0.173, 0.345, 0.476, 0.582, 0.704, 0.856, 1.0])
# left / right value columns of the upper field grid (fractions of grid width)
GRID_L = (0.292, 0.516)
GRID_R = (0.778, 1.0)


# ----------------------------------------------------------------------------------------- loading
def load_image(src, hw=HW):
    """path | PIL | ndarray -> uint8 RGB ndarray (EXIF-upright), about `hw` px wide (never upscaled beyond 1.0)."""
    if isinstance(src, np.ndarray):
        im = Image.fromarray(src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2RGB))
    elif isinstance(src, Image.Image):
        im = src
    else:
        p = str(src)
        if p.lower().endswith(".pdf"):
            import pymupdf
            with pymupdf.open(p) as d:
                pg = d[0]
                im = Image.open(io.BytesIO(pg.get_pixmap(dpi=int(72 * hw / pg.rect.width)).tobytes("png")))
        else:
            im = Image.open(p)
            try:
                im.draft("RGB", (hw, hw))
            except Exception:
                pass
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width > hw * 1.05:
        im = im.resize((hw, int(round(im.height * hw / im.width))), Image.BILINEAR)
    return np.asarray(im)


def _gray(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _resize_w(a, w):
    s = w / a.shape[1]
    return cv2.resize(a, (w, max(1, int(round(a.shape[0] * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


def _binar(g, block=21, c=7):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c)


# ----------------------------------------------------------------------------------------- orientation
def angle_orient(g):
    """Skew (deg, in (-45,45]) of the dominant ruled lines + whether horizontal lines dominate. None if no lines."""
    sm = _resize_w(g, 500)
    b = _binar(sm)
    ls = cv2.HoughLinesP(b, 1, np.pi / 720, threshold=60, minLineLength=90, maxLineGap=10)
    if ls is None:
        return None
    ls = ls.reshape(-1, 4).astype(float)
    dx = ls[:, 2] - ls[:, 0]
    dy = ls[:, 3] - ls[:, 1]
    L = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 180
    h = np.zeros(360)
    np.add.at(h, (ang * 2).astype(int) % 360, L)
    hs = np.convolve(np.r_[h[-6:], h, h[:6]], np.ones(5), "same")[6:-6]
    d = -45 + 0.5 * np.arange(180)
    hi = ((d * 2) % 360).astype(int)
    vi = (((d + 90) * 2) % 360).astype(int)
    tot = hs[hi] + hs[vi]
    i = int(np.argmax(tot))
    return float(d[i]), bool(hs[hi[i]] >= hs[vi[i]]), float(hs[hi[i]]), float(hs[vi[i]])


def rotate_small(a, deg, fill=255):
    if abs(deg) < 0.15:
        return a
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(fill,) * (3 if a.ndim == 3 else 1))


def find_barcode(g):
    """Locate the printed barcode (vertical bars): returns (x0,y0,x1,y1,score) in g coordinates or None."""
    H, W = g.shape
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    k = (max(9, W // 25), max(5, W // 70))
    sx = cv2.boxFilter(gx, -1, k)
    sy = cv2.boxFilter(gy, -1, k)
    sc = sx - 2.2 * sy
    sc[:, : int(0.3 * W)] = -1e9
    sc[int(0.4 * H):] = -1e9
    y, x = np.unravel_index(int(np.argmax(sc)), sc.shape)
    s = float(sc[y, x])
    if s < 15:
        return s, None
    m = (sx - 2.2 * sy) > 0.45 * s
    m = m.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    l = lab[y, x]
    if l == 0:
        return s, None
    x0, y0, w, h = st[l, :4]
    return s, (int(x0 - k[0] // 2), int(y0 - k[1] // 2), int(x0 + w + k[0] // 2), int(y0 + h + k[1] // 2))


def normalise(rgb):
    """Rotate (90/180/270) + deskew so the page is upright. Returns (rgb_upright, info dict)."""
    g = _gray(rgb)
    info = {"rot": 0, "skew": 0.0, "barcode_score": 0.0, "lines": False}
    ao = angle_orient(_resize_w(g, GW))
    if ao is None:
        return rgb, info
    d, horiz, hh, vv = ao
    info.update(lines=True, skew=d)
    base = rotate_small(rgb, d)
    cands = [0, 2] if horiz else [1, 3]
    best = None
    scores = []
    for k in cands:
        r = np.ascontiguousarray(np.rot90(base, k))
        gg = _resize_w(_gray(r), GW)
        s, bb = find_barcode(gg)
        # barcode should be in the top-right and have a sane size
        if bb is not None:
            W = gg.shape[1]
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            if not (0.08 * W < bw < 0.4 * W and bh < 0.12 * W):
                s *= 0.3
        scores.append(s)
        if best is None or s > best[0]:
            best = (s, k, r)
    info["barcode_score"] = best[0]
    info["barcode_scores"] = scores
    info["rot"] = best[1] * 90
    r = best[2]
    # second, finer deskew pass on the upright image
    for _ in range(1):
        a2 = refine_skew(_resize_w(_gray(r), GW))
        if abs(a2) >= 0.15:
            r = rotate_small(r, a2)
            info["skew"] += a2
    return r, info


# ----------------------------------------------------------------------------------------- ruled-line structure
def _peaks1d(prof, thr, merge):
    """centres of runs where prof>thr (runs closer than `merge` are joined); returns [(centre, strength)]"""
    idx = np.where(prof > thr)[0]
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > merge:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    res = []
    for c in out:
        w = prof[c[0]: c[-1] + 1]
        res.append((float((np.arange(c[0], c[-1] + 1) * w).sum() / max(w.sum(), 1e-9)), float(w.max())))
    return res


def line_masks(g, relax=False):
    """g: 1000-wide upright gray. -> (None, hl, vl) using black-hat (works on faint ruled lines); relax=True for
    blurry / low-contrast photos (thicker lines, lower thresholds)."""
    W = g.shape[1]
    kk = 15 if relax else 9
    if relax:
        g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    bh_h = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kk)))
    bh_v = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kk, 1)))
    hl = cv2.morphologyEx(bh_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * (0.04 if relax else 0.05)), 1)))
    vl = cv2.morphologyEx(bh_v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(W * (0.025 if relax else 0.03)))))
    th = max(4.0 if relax else 8.0, (0.15 if relax else 0.3) * float(np.percentile(hl, 99.7)))
    tv = max(4.0 if relax else 8.0, (0.15 if relax else 0.3) * float(np.percentile(vl, 99.7)))
    hl = ((hl > th) * 255).astype(np.uint8)
    vl = ((vl > tv) * 255).astype(np.uint8)
    return None, hl, vl


def refine_skew(g):
    """residual skew (deg) of an already roughly upright 1000 px gray image: projection-profile search on the
    ruled-line mask (sharpest row histogram wins)."""
    _, hl, _ = line_masks(g)
    sm = cv2.resize(hl, (500, int(hl.shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
    h, w = sm.shape

    def score(a):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(sm, M, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)
        prof = r.sum(1)
        return float((prof ** 2).sum())

    angs = np.arange(-4.0, 4.01, 0.5)
    sc = [score(a) for a in angs]
    a0 = angs[int(np.argmax(sc))]
    angs2 = np.arange(a0 - 0.5, a0 + 0.51, 0.1)
    sc2 = [score(a) for a in angs2]
    return float(angs2[int(np.argmax(sc2))])


def hlines(hl, x0=0, x1=None, thr_frac=0.3):
    """y positions of long horizontal ruled lines: [(y, strength_px)]"""
    W = hl.shape[1]
    x1 = x1 or W
    prof = (hl[:, x0:x1] > 0).sum(1).astype(float)
    prof = prof + np.r_[prof[1:], 0] + np.r_[0, prof[:-1]]     # tolerate 1-2 px slant
    return _peaks1d(prof, thr_frac * W, 3)


def fit_row_chain(lines, W):
    """Find the arithmetic chain of equally spaced lines = rows of the lower table.
    -> dict(y0, p, n_hit, idx_hits) where y(i)=y0+i*p is the TOP of row 1 (i=0) .. bottom of row 10 (i=10)"""
    ys = np.array([l[0] for l in lines])
    if len(ys) < 5:
        return None
    best = None
    dif = np.diff(ys)
    cand_p = [d for d in dif if 0.014 * W < d < 0.042 * W]
    if not cand_p:
        return None
    for p0 in sorted(set(np.round(cand_p, 0)))[:40]:
        for s in range(len(ys)):
            hits = [s]
            y = ys[s]
            miss = 0
            k = 1
            while True:
                t = ys[s] + k * p0
                j = np.where(np.abs(ys - t) < 0.22 * p0)[0]
                if len(j):
                    hits.append(int(j[0]))
                    miss = 0
                else:
                    miss += 1
                    if miss > 2:
                        break
                k += 1
                if k > 14:
                    break
            nh = len(hits)
            if best is None or nh > best[0] or (nh == best[0] and s > best[2]):
                best = (nh, p0, s, hits)
    if best is None or best[0] < 8:
        return None
    nh, p0, s, hits = best
    hy = ys[hits]
    # LSQ fit y = a + b*k with integer k from rounding
    k = np.round((hy - hy[0]) / p0).astype(int)
    A = np.vstack([np.ones_like(k), k]).T
    a, b = np.linalg.lstsq(A, hy, rcond=None)[0]
    return {"y_first": float(a), "p": float(b), "k_hits": k.tolist(), "n_hit": nh}


def _col_fit(xs_strength, W):
    """fit (L, R) of the template column grid to the detected vertical lines: any two peaks may be assigned to any two
    template columns. -> (L, R, n_matched) or None"""
    xs = np.array([x for x, _ in xs_strength])
    if len(xs) < 3:
        return None
    best = None
    fr = COL_FR
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            for a in range(len(fr)):
                for b in range(a + 1, len(fr)):
                    w = (xs[j] - xs[i]) / (fr[b] - fr[a])
                    if not (0.55 * W < w < 0.98 * W):
                        continue
                    L = xs[i] - fr[a] * w
                    pred = L + fr * w
                    tol = 0.02 * w
                    err = np.array([np.min(np.abs(xs - q)) for q in pred])
                    m = int((err < tol).sum())
                    sc_ = m - 0.5 * float(np.minimum(err, tol).mean()) / tol
                    if best is None or sc_ > best[3]:
                        best = (float(L), float(L + w), m, sc_)
    return best[:3] if best else None


def line_extent(hl, y, gap):
    """(x0, x1) of the longest horizontal run (gaps <= gap px bridged) of the ruled-line mask around row y, or None"""
    H, W = hl.shape
    y0, y1 = int(max(0, round(y) - 3)), int(min(H, round(y) + 4))
    if y1 <= y0:
        return None
    row = (hl[y0:y1] > 0).any(0)
    xs = np.where(row)[0]
    if len(xs) == 0:
        return None
    best, start, prev = None, xs[0], xs[0]
    for x in xs[1:]:
        if x - prev > gap:
            if best is None or prev - start > best[1] - best[0]:
                best = (start, prev)
            start = x
        prev = x
    if best is None or prev - start > best[1] - best[0]:
        best = (start, prev)
    return (float(best[0]), float(best[1]))


def locate_table(g, hl, vl):
    """Lower table of Proforma-3. g: 1000-wide upright gray. Returns dict or None:
       y0 (top of row 1), p (row pitch), ytop (header top), ybot_rows (bottom of row 10), ytot (bottom of total row),
       xs (9 column edges at the rows band), L, R, n_cols_matched, n_row_lines"""
    H, W = g.shape
    lines = hlines(hl, thr_frac=0.22)
    ch = fit_row_chain(lines, W)
    if ch is None:
        return None
    p = ch["p"]
    ys = np.array([l[0] for l in lines])
    k0 = ch["k_hits"][0]
    y_first = ch["y_first"]
    # chain line positions
    chain_y = [y_first + k * p for k in range(0, max(ch["k_hits"]) + 1)]
    # the top of row 1 is the chain line under which the header (about 1.93p) lies
    y0 = None
    for yc in chain_y[:5]:
        if not np.any(np.abs(ys - yc) < 0.25 * p):
            continue
        if np.any(np.abs(ys - (yc - 1.93 * p)) < 0.3 * p):
            y0 = yc
            break
    if y0 is None:
        y0 = chain_y[0]
    # column lines inside the rows band
    ya, yb = int(y0 + 0.2 * p), int(min(H - 1, y0 + 9.8 * p))
    if yb <= ya:
        return None
    vprof = (vl[ya:yb, :] > 0).sum(0).astype(float)
    vprof = vprof + np.r_[vprof[1:], 0] + np.r_[0, vprof[:-1]]
    vx = _peaks1d(vprof, 0.25 * (yb - ya), 3)
    vxa = np.array([x for x, _ in vx]) if vx else np.zeros(0)

    def nmatch(L_, R_):
        w_ = R_ - L_
        if w_ <= 0:
            return 0
        pred = L_ + COL_FR * w_
        return sum(1 for q in pred if len(vxa) and np.min(np.abs(vxa - q)) < 0.02 * w_)

    cands = []
    cf = _col_fit(vx, W)
    if cf is not None:
        cands.append((cf[0], cf[1], cf[2], "vlines"))
    # extents of the full-width horizontal lines (header top, row-10 bottom, total bottom)
    ext = []
    for yy in (y0 - 1.93 * p, y0 + 10 * p, y0 + 10.77 * p):
        seg = line_extent(hl, yy, 0.012 * W)
        if seg is not None and seg[1] - seg[0] > 0.5 * W:
            ext.append(seg)
    if len(ext) >= 1:
        Le = float(np.median([e[0] for e in ext]))
        Re = float(np.median([e[1] for e in ext]))
        cands.append((Le, Re, nmatch(Le, Re), "extent"))
        if len(ext) >= 2:
            # also the widest line alone (others may be partial)
            e = max(ext, key=lambda t: t[1] - t[0])
            cands.append((e[0], e[1], nmatch(e[0], e[1]), "widest"))
    if not cands:
        return None
    cands.sort(key=lambda c: (c[2], c[3] == "extent"), reverse=True)
    L, R, m, how = cands[0]
    if m < 4:
        return None
    w = R - L
    xs = L + COL_FR * w
    ytop = y0 - 1.93 * p
    ybot = y0 + 10 * p
    ytot = ybot + 0.77 * p
    # snap bottom / total lines to detected ones
    for name, yy in (("ybot", ybot), ("ytot", ytot)):
        j = np.where(np.abs(ys - yy) < 0.3 * p)[0]
        if len(j):
            yy = float(ys[j[np.argmin(np.abs(ys[j] - yy))]])
        if name == "ybot":
            ybot = yy
        else:
            ytot = yy
    if ytot - ybot < 0.45 * p:
        ytot = ybot + 0.77 * p
    return {"y0": float(y0), "p": float(p), "ytop": float(ytop), "ybot": float(ybot), "ytot": float(ytot), "xs": xs.tolist(),
            "L": L, "R": R, "n_cols": int(m), "n_rowlines": int(ch["n_hit"]), "col_src": how}


def snap_cols(vl, xs, y0, y1, w):
    """re-locate each predicted column edge inside the horizontal band [y0,y1] (handles keystone / residual skew)"""
    H, W = vl.shape
    y0, y1 = int(max(0, y0)), int(min(H, y1))
    out = []
    if y1 - y0 < 5:
        return list(xs), 0
    band = (vl[y0:y1] > 0).sum(0).astype(float)
    band = band + np.r_[band[1:], 0] + np.r_[0, band[:-1]]
    tol = int(0.022 * w)
    n = 0
    for x in xs:
        a, b = int(max(0, x - tol)), int(min(W, x + tol + 1))
        seg = band[a:b]
        if len(seg) and seg.max() >= 0.45 * (y1 - y0):
            out.append(float(a + np.argmax(seg)))
            n += 1
        else:
            out.append(float(x))
    return out, n


def cell_boxes(tab, vl, scale):
    """boxes (x0,y0,x1,y1) in hi-res coords for area / loss cells of rows 1,2 and of the total row."""
    p, y0 = tab["p"], tab["y0"]
    w = tab["R"] - tab["L"]
    out = {}
    bands = {"r1": (y0, y0 + p), "r2": (y0 + p, y0 + 2 * p), "tot": (tab["ybot"], tab["ytot"])}
    for name, (ya, yb) in bands.items():
        xs, n = snap_cols(vl, tab["xs"], ya + 0.1 * (yb - ya), yb - 0.1 * (yb - ya), w) if name != "tot" else snap_cols(vl, tab["xs"], ya + 2, yb - 2, w)
        if name == "tot":
            # the total-row band is short: fall back to the row band snap for columns that did not snap
            xr, _ = snap_cols(vl, tab["xs"], y0 + 2 * p, tab["ybot"], w)
            xs = [a if abs(a - b) > 0.001 else c for a, b, c in zip(xs, tab["xs"], xr)]
        for cname, (i0, i1) in {"area": (5, 6), "loss": (6, 7)}.items():
            padv = 0.08 * (yb - ya)
            out[f"{cname}_{name}"] = tuple(int(round(v * scale)) for v in (xs[i0], ya - padv, xs[i1], yb + padv))
    return out


GRID_H = 11.6       # grid height in row pitches (template constant)
GRID_GAP = 6.8      # header-top of table to grid bottom, in pitches


def grid_lines(hl, tab, W):
    """Upper field grid (fields 1-15). Returns dict(lines(top->bottom), B, detected) in g coords; falls back to nominal."""
    L, R, p = tab["L"], tab["R"], tab["p"]
    w = R - L
    ls = hlines(hl, int(L - 0.03 * w), int(R + 0.03 * w), thr_frac=0.1)
    cand = [y for y, s in ls if s >= 0.3 * w and y < tab["ytop"] - 0.9 * p]
    cand.sort(reverse=True)
    nomB = tab["ytop"] - GRID_GAP * p
    for i, B in enumerate(cand):
        if abs(B - nomB) > 1.6 * p:
            continue
        chain = [B]
        for y in cand[i + 1:]:
            d = chain[-1] - y
            if d < 0.8 * p:
                continue
            if d > 2.6 * p:
                break
            chain.append(y)
        if len(chain) >= 4:
            return {"lines": chain[::-1], "B": B, "detected": True}
    return {"lines": [nomB], "B": nomB, "detected": False}


def grid_bands(gr, tab):
    """rows 11/12, 13/14, 15 of the grid as (ya, yb) + grid top T"""
    p = tab["p"]
    ln = gr["lines"]
    B = gr["B"]
    out = {}
    nom = {"r15": 1.16, "r1314": 1.94, "r1112": 1.14}
    cur = B
    for nm in ("r15", "r1314", "r1112"):
        exp = cur - nom[nm] * p
        j = [y for y in ln if abs(y - exp) < 0.32 * p]
        nxt = min(j, key=lambda y: abs(y - exp)) if j else exp
        out[nm] = (nxt, cur)
        cur = nxt
    Tn = B - GRID_H * p
    j = [y for y in ln if abs(y - Tn) < 0.6 * p]
    out["T"] = min(j, key=lambda y: abs(y - Tn)) if j else Tn
    out["detected"] = gr["detected"]
    return out


class Layout(dict):
    """dict with: ok, rgb (normalised hi-res image), scale, tab, boxes{name:(x0,y0,x1,y1) in hi px}, notes"""


def _layout(r, info):
    lay = Layout(ok=False, notes=[])
    lay.update(rgb=r, info=info)
    g = _resize_w(_gray(r), GW)
    sc = r.shape[1] / GW
    lay["scale"] = sc
    lay["g"] = g
    if not info["lines"]:
        lay["notes"].append("no ruled lines found")
        return lay
    _, hl, vl = line_masks(g)
    tab = locate_table(g, hl, vl)
    if tab is None:          # blurry / low contrast: retry with relaxed line extraction
        _, hl, vl = line_masks(g, relax=True)
        tab = locate_table(g, hl, vl)
        if tab is not None:
            lay["notes"].append("low-contrast page (relaxed line detection)")
    lay["hl"], lay["vl"] = hl, vl
    if tab is None:
        lay["notes"].append("table not located")
        return lay
    lay["tab"] = tab
    lay["ok"] = True
    p, w, L, R = tab["p"], tab["R"] - tab["L"], tab["L"], tab["R"]
    H = g.shape[0]
    B = {}
    # table cells
    B.update(cell_boxes(tab, vl, sc))
    # upper grid -> dates, PO ID
    gr = grid_lines(hl, tab, GW)
    lay["grid"] = gr
    if gr is not None:
        gb = grid_bands(gr, tab)
        if not gb["detected"]:
            lay["notes"].append("upper grid lines not detected (nominal layout used)")
        lay["gb"] = gb
        for nm, (a, b) in (("r1112", gb["r1112"]), ("r1314", gb["r1314"])):
            xs, _ = snap_cols(vl, [L + f * w for f in (GRID_L[0], GRID_L[1], GRID_R[0], 1.0)], a + 0.15 * p, b - 0.15 * p, w)
            padv = 0.06 * (b - a)
            B[f"{'sow' if nm == 'r1112' else 'intim'}_date"] = tuple(int(round(v * sc)) for v in (xs[0] + 2, a - padv, xs[1] - 2, b + padv))
            B[f"{'loss' if nm == 'r1112' else 'insp'}_date"] = tuple(int(round(v * sc)) for v in (xs[2] + 2, a - padv, xs[3] + 0.02 * w, b + padv))
        T = gb["T"]
        B["po_id"] = tuple(int(round(v * sc)) for v in (L + 0.58 * w, T - 2.3 * p, R + 0.02 * w, T + 0.1 * p))
        B["formno_win"] = tuple(int(round(v * sc)) for v in (R - 0.34 * w, T - 5.6 * p, R + 0.04 * w, T - 2.4 * p))
    else:
        lay["notes"].append("upper grid not located")
    # signatures
    yt = tab["ytot"]
    fr = [(0.0, 0.235), (0.232, 0.49), (0.494, 0.745), (0.748, 1.0)]
    for nm, (a, b) in zip(("farmer", "company", "worker", "officer"), fr):
        B[f"sig_{nm}"] = tuple(int(round(v * sc)) for v in (L + a * w, yt + 1.5 * p, L + b * w, yt + 3.55 * p))
        B[f"lab_{nm}"] = tuple(int(round(v * sc)) for v in (L + a * w, yt + 3.35 * p, L + b * w, yt + 4.6 * p))
    lay["boxes"] = B
    return lay


def analyse_layout(src_rgb):
    """Full geometric analysis of one photographed page. Returns Layout (ok False when the ruled table is not found)."""
    r, info = normalise(src_rgb)
    lay = _layout(r, info)
    sc = info.get("barcode_scores") or []
    sure = len(sc) == 2 and max(sc) >= 40 and max(sc) >= 2.0 * max(1e-6, min(sc)) and lay["ok"]
    if info.get("lines") and not sure and len(sc) == 2:
        # orientation is ambiguous (barcode not conclusive): also try the page turned by 180 degrees and keep the
        # reading in which the ruled table has the upper field grid above it
        r2 = np.ascontiguousarray(r[::-1, ::-1])
        info2 = dict(info, rot=(info["rot"] + 180) % 360)
        lay2 = _layout(r2, info2)

        def score(l):
            if not l["ok"]:
                return -1
            g = l.get("grid")
            return 10 * bool(g and g.get("detected")) + l["tab"]["n_cols"] + 0.1 * l["tab"]["n_rowlines"]
        def pos(l):         # table should sit in the lower half of an upright page
            return 0.0 if not l["ok"] else (1.0 if l["tab"]["y0"] > 0.42 * l["g"].shape[0] else -1.0)
        if (score(lay2) + 3 * pos(lay2)) > (score(lay) + 3 * pos(lay)) + 0.5:
            lay = lay2
            lay["notes"].append("orientation decided by table/grid structure")
    return lay


def crop(rgb, box, pad=0):
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(w, x1 + pad)), int(min(h, y1 + pad))
    return rgb[y0:y1, x0:x1]


_BD = None


def find_barcode_quad(r, lay):
    """Barcode quad (4x2 float array, order BL,TL,TR,BR) in hi-res coords using OpenCV's detector inside the
    window above the upper grid; None when not found."""
    global _BD
    if _BD is None:
        _BD = cv2.barcode.BarcodeDetector()
    B = lay["boxes"]
    if "formno_win" not in B:
        return None
    x0, y0, x1, y1 = B["formno_win"]
    p = lay["tab"]["p"] * lay["scale"]
    wx0, wx1 = int(max(0, x0 - 0.1 * (x1 - x0))), int(min(r.shape[1], x1 + 0.15 * (x1 - x0)))
    wy0, wy1 = int(max(0, y0 - 1.2 * p)), int(min(r.shape[0], y1 + 3.5 * p))
    g = cv2.cvtColor(r[wy0:wy1, wx0:wx1], cv2.COLOR_RGB2GRAY)
    try:
        res = _BD.detectAndDecodeWithType(g)
    except Exception:
        return None
    pts = res[3]
    if pts is None or len(pts) == 0:
        return None
    best = None
    for q in np.asarray(pts).reshape(-1, 4, 2):
        w = np.linalg.norm(q[3] - q[0])
        h = np.linalg.norm(q[0] - q[1])
        if w < 0.12 * r.shape[1] or h < 8 or w / h < 3:
            continue
        if best is None or w > best[0]:
            best = (w, q)
    if best is None:
        return None
    q = best[1] + np.array([wx0, wy0], np.float32)
    return q


def formno_strip(r, q, scale=3.0, lo=0.04, hi=0.95):
    """rectified strip under the barcode quad: gray uint8"""
    p0, p1, p2, p3 = q
    d = (p0 - p1)             # down vector, length = barcode height
    top_l, top_r = p0 + lo * d, p3 + lo * d
    bot_l, bot_r = p0 + hi * d, p3 + hi * d
    bw = np.linalg.norm(p3 - p0)
    bh = np.linalg.norm(d)
    wd, hd = int(bw * scale), int(bh * (hi - lo) * scale)
    src = np.float32([top_l, top_r, bot_r, bot_l])
    dst = np.float32([[0, 0], [wd, 0], [wd, hd], [0, hd]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), M, (wd, hd), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------------------- ink measurements
def ink_mask(rgb_or_gray, min_len_frac=0.3, contrast=18):
    """binary handwriting-like ink mask (uint8 0/255): pixels darker than their local background, straight ruled
    lines removed. Works on gray or RGB crops."""
    a = np.asarray(rgb_or_gray)
    g = _gray(a) if a.ndim == 3 else a
    if g.size == 0:
        return np.zeros((1, 1), np.uint8)
    h, w = g.shape
    k = max(3, (min(h, w) // 2) | 1)
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), 2)
    d = bg.astype(np.int16) - g.astype(np.int16)
    b = (d > contrast).astype(np.uint8) * 255
    hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * min_len_frac)), 1)))
    vl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.6)))))
    hl = cv2.dilate(hl, np.ones((3, 1), np.uint8))
    vl = cv2.dilate(vl, np.ones((1, 3), np.uint8))
    return cv2.subtract(b, cv2.bitwise_or(hl, vl))


def ink_frac(crop, inset=0.06):
    """fraction of ink pixels in the crop (borders inset to ignore ruled-line remains)"""
    m = ink_mask(crop)
    h, w = m.shape
    dy, dx = int(h * inset), int(w * inset)
    m = m[dy: h - dy, dx: w - dx]
    return float((m > 0).mean()) if m.size else 0.0
