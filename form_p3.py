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
        if best is None or s > best[0]:
            best = (s, k, r)
    info["barcode_score"] = best[0]
    info["rot"] = best[1] * 90
    r = best[2]
    # second, finer deskew pass on the upright image
    for _ in range(2):
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


def line_masks(g):
    """g: 1000-wide upright gray. -> (binary-ish ink, hl, vl) using black-hat (works on faint ruled lines)."""
    W = g.shape[1]
    bh_h = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9)))
    bh_v = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1)))
    hl = cv2.morphologyEx(bh_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.05), 1)))
    vl = cv2.morphologyEx(bh_v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(W * 0.03))))
    th = max(8.0, 0.3 * float(np.percentile(hl, 99.7)))
    tv = max(8.0, 0.3 * float(np.percentile(vl, 99.7)))
    hl = ((hl > th) * 255).astype(np.uint8)
    vl = ((vl > tv) * 255).astype(np.uint8)
    return None, hl, vl


def refine_skew(g):
    """small residual skew (deg) from long horizontal ruled lines of an already roughly upright 1000 px gray image"""
    _, hl, _ = line_masks(g)
    ls = cv2.HoughLinesP(hl, 1, np.pi / 1440, 120, minLineLength=int(0.25 * g.shape[1]), maxLineGap=15)
    if ls is None:
        return 0.0
    ang, wt = [], []
    for x1, y1, x2, y2 in ls.reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 8:
            ang.append(a)
            wt.append(np.hypot(x2 - x1, y2 - y1))
    if not ang:
        return 0.0
    o = np.argsort(ang)
    ang, wt = np.array(ang)[o], np.array(wt)[o]
    c = np.cumsum(wt)
    return float(ang[np.searchsorted(c, c[-1] / 2)])


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
    """choose (L,R) among vertical-line x candidates maximising matches to COL_FR. -> (L,R,matches) or None"""
    xs = np.array([x for x, _ in xs_strength])
    if len(xs) < 4:
        return None
    best = None
    for i in range(len(xs)):
        for j in range(i + 3, len(xs)):
            L, R = xs[i], xs[j]
            w = R - L
            if not (0.55 * W < w < 0.98 * W):
                continue
            pred = L + COL_FR * w
            tol = 0.012 * w
            m = sum(1 for q in pred if np.min(np.abs(xs - q)) < tol)
            if best is None or m > best[2] or (m == best[2] and w > best[1] - best[0]):
                best = (float(L), float(R), m)
    return best


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
    cf = _col_fit(vx, W)
    if cf is None:
        return None
    L, R, m = cf
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
            "L": L, "R": R, "n_cols": int(m), "n_rowlines": int(ch["n_hit"])}


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
