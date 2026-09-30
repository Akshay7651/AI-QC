"""Self-contained locator for the Proforma-3 table (used to build training/test crops for digits.py).

locate_table(img) -> dict(gray=deskewed 1600-wide gray image, hlines=[y..], vlines=[x..], rows=[(y0,y1)..], cols=[(x0,x1)..]) or None
cell_crops(img) -> dict(area=[row1..], loss=[row1..], total_area, total_loss, ...)  PIL 'L' crops (ruled lines left in)
"""
import cv2
import numpy as np
from PIL import Image

W0 = 1600


def _gray(img):
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("L"))
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _bin(g):
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 12)


def _skew(b):
    hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1)))
    ls = cv2.HoughLinesP(hl, 1, np.pi / 720, 200, minLineLength=300, maxLineGap=20)
    if ls is None:
        return 0.0
    ang, wt = [], []
    for x1, y1, x2, y2 in np.asarray(ls).reshape(-1, 4):
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(a) < 15:
            ang.append(a)
            wt.append(np.hypot(x2 - x1, y2 - y1))
    if not ang:
        return 0.0
    o = np.argsort(ang)
    ang, wt = np.array(ang)[o], np.array(wt)[o]
    c = np.cumsum(wt)
    return float(ang[np.searchsorted(c, c[-1] / 2)])


def _peaks(prof, thr, gap):
    idx = np.where(prof > thr)[0]
    out, cur = [], []
    for i in idx:
        if cur and i - cur[-1] > gap:
            out.append(int(np.mean(cur)))
            cur = []
        cur.append(i)
    if cur:
        out.append(int(np.mean(cur)))
    return out


def locate_table(img):
    g = _gray(img)
    if g.shape[0] < g.shape[1]:  # landscape -> try rotating (forms are portrait)
        g = cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)
    h0, w0 = g.shape
    g = cv2.resize(g, (W0, int(h0 * W0 / w0)), interpolation=cv2.INTER_AREA)
    for _ in range(2):
        b = _bin(g)
        a = _skew(b)
        if abs(a) > 0.05:
            H, W = g.shape
            M = cv2.getRotationMatrix2D((W / 2, H / 2), a, 1.0)
            g = cv2.warpAffine(g, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=int(np.median(g)))
    H, W = g.shape
    b = _bin(g)
    hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (90, 1)))
    # long horizontal lines
    prof = hl.sum(1) / 255.0
    ys = _peaks(prof, 0.30 * W * 0.5, 3)
    ys = [y for y in ys if y > 0.3 * H]
    if len(ys) < 8:
        return None
    # find a run of >=11 roughly equally spaced lines (header bottom .. total row)
    best = None
    for i in range(len(ys)):
        run = [ys[i]]
        for y in ys[i + 1:]:
            d = y - run[-1]
            if d < 14:
                continue
            if len(run) >= 2 and not (0.7 * np.median(np.diff(run)) < d < 1.4 * np.median(np.diff(run))):
                if d > 1.4 * np.median(np.diff(run)):
                    break
                continue
            run.append(y)
        if best is None or len(run) > len(best):
            best = run
    if best is None or len(best) < 9:
        return None
    hlines = best
    y0, y1 = hlines[0] - 60, hlines[-1] + 10
    y0 = max(0, y0)
    vl = cv2.morphologyEx(b[y0:y1], cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60)))
    xs = _peaks(vl.sum(0) / 255.0, 120, 6)
    return dict(gray=g, hlines=hlines, vlines=xs, y0=y0)


def debug(img, path):
    t = locate_table(img)
    if t is None:
        return False
    c = cv2.cvtColor(t["gray"], cv2.COLOR_GRAY2BGR)
    for y in t["hlines"]:
        cv2.line(c, (0, y), (c.shape[1], y), (0, 0, 255), 1)
    for x in t["vlines"]:
        cv2.line(c, (x, t["hlines"][0]), (x, t["hlines"][-1]), (0, 255, 0), 1)
    cv2.imwrite(path, c[max(0, t["hlines"][0] - 200): t["hlines"][-1] + 100])
    return True


# ---------------------------------------------------------------- PO ID strip
def po_strip(t):
    """t = locate_table() result -> gray crop of the handwritten PO ID line (right half, above the 1st table)."""
    g, hl = t["gray"], t["hlines"]
    H, W = g.shape
    exp = hl[0] - 920
    b = _bin(g[: int(0.45 * H)])
    hm = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (260, 1)))
    ys = _peaks(hm.sum(1) / 255.0, 240, 5)
    c = [y for y in ys if abs(y - exp) < 110]
    y = min(c, key=lambda v: abs(v - exp)) if c else exp
    y = int(y)
    return g[max(0, y - 80): y + 10, int(0.50 * W):]


# ---------------------------------------------------------------- cells
PRIOR_V = (924.0, 1084.0, 1280.0)  # left edge of affected-area col, loss col, remarks col (1600-wide page)


def columns(vlines):
    """Snap the three prior column edges to detected vertical lines; shift by mean offset if only some found."""
    found, offs = [], []
    for p in PRIOR_V:
        c = [x for x in vlines if abs(x - p) < 55]
        found.append(min(c, key=lambda x: abs(x - p)) if c else None)
        if c:
            offs.append(found[-1] - p)
    sh = float(np.mean(offs)) if offs else 0.0
    return [f if f is not None else p + sh for f, p in zip(found, PRIOR_V)], len(offs)


def cell_crops(img, with_total=True):
    """Return dict with keys area, loss (row 1) and t_area, t_loss (total row) as PIL 'L' crops, plus 'ok' flags."""
    t = locate_table(img)
    if t is None:
        return None
    g, hl = t["gray"], t["hlines"]
    (xa, xl, xr), nfound = columns(t["vlines"])
    pitch = float(np.median(np.diff(hl)))
    rows = {"": (hl[0], hl[0] + pitch)}
    if len(hl) >= 12:
        rows["t_"] = (hl[-2], hl[-1])
    out = {"nfound": nfound, "pitch": pitch}
    H, W = g.shape
    for pre, (ya, yb) in rows.items():
        ya, yb = int(ya), int(yb)
        for name, (xa_, xb_) in (("area", (xa, xl)), ("loss", (xl, xr))):
            out[pre + name] = Image.fromarray(g[max(0, ya - 6): min(H, yb + 6), int(xa_) + 3: int(xb_) - 3])
    return out
