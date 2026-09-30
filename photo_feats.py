"""Cheap image statistics for evidence photos (all on a ~160x232 downscaled copy of the content area)."""
import cv2
import numpy as np

W, H = 160, 232
STAMP_FRAC = 0.82          # bottom 18% holds the burned-in stamp - excluded from content statistics


def content(bgr):
    h = bgr.shape[0]
    c = bgr[: int(h * STAMP_FRAC)]
    return cv2.resize(c, (W, H), interpolation=cv2.INTER_AREA)


def _runs(b, axis=1):
    """Fraction of pixels lying in long straight runs (ruled lines / table borders)."""
    k = (25, 1) if axis == 1 else (1, 25)
    op = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
    return float((op > 0).mean()), op


FORM_NAMES = None


def form_features(c):
    """c: BGR uint8 (H x W). Returns dict of named floats describing 'paper form / document' likeness."""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    s, v, hue = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32), hsv[..., 0].astype(np.float32)
    f = {}
    f["sat_mean"] = s.mean() / 255
    f["lowsat"] = float((s < 40).mean())
    paper = (s < 55) & (v > 150)
    f["paper"] = float(paper.mean())
    # largest connected pale region
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(paper.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    f["paper_cc"] = float(st[1:, 4].max() / (W * H)) if n > 1 else 0.0
    exg = 2 * c[..., 1].astype(np.float32) - c[..., 0] - c[..., 2]
    f["green"] = float((exg > 25).mean())
    f["green_max_block"] = float(max((exg[i * H // 3:(i + 1) * H // 3, j * W // 3:(j + 1) * W // 3] > 25).mean() for i in range(3) for j in range(3)))
    f["red_ink"] = float((((hue < 8) | (hue > 170)) & (s > 90) & (v > 80)).mean())
    f["blue_ink"] = float(((hue > 100) & (hue < 135) & (s > 70) & (v < 200)).mean())
    f["v_std"] = float(v.std() / 255)
    # local contrast and text-like structure: adaptive binarisation
    gb = cv2.GaussianBlur(g, (3, 3), 0)
    b = cv2.adaptiveThreshold(gb, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 8)
    f["dark_frac"] = float((b > 0).mean())
    f["hrun"], hop = _runs(b, 1)
    f["vrun"], vop = _runs(b, 0)
    # rows with a long horizontal line
    f["hline_rows"] = float((hop.sum(axis=1) > 255 * 40).sum() / H)
    f["vline_cols"] = float((vop.sum(axis=0) > 255 * 40).sum() / W)
    # row profile of dark pixels: text lines make a strongly periodic profile
    rp = (b > 0).mean(axis=1).astype(np.float32)
    rp = rp - rp.mean()
    ac = np.correlate(rp, rp, "full")[H - 1:]
    ac = ac / (ac[0] + 1e-6)
    f["rowac_peak"] = float(ac[4:30].max())
    f["rowprof_cv"] = float(rp.std() / (np.mean((b > 0).mean(axis=1)) + 1e-6))
    cp = (b > 0).mean(axis=0).astype(np.float32)
    f["colprof_cv"] = float(cp.std() / (cp.mean() + 1e-6))
    # small connected dark components (characters) density
    n2, _, st2, _ = cv2.connectedComponentsWithStats(b)
    a = st2[1:, 4]; wh = st2[1:, 2:4]
    charlike = (a >= 3) & (a <= 80) & (wh.max(axis=1) <= 14)
    f["char_density"] = float(charlike.sum() / (W * H) * 100)
    f["big_cc"] = float((a > 400).sum())
    # edges
    ed = cv2.Canny(gb, 60, 150)
    f["edge"] = float((ed > 0).mean())
    gx = cv2.Sobel(gb, cv2.CV_32F, 1, 0); gy = cv2.Sobel(gb, cv2.CV_32F, 0, 1)
    ex, ey = float(np.abs(gx).mean()), float(np.abs(gy).mean())
    f["edge_xy"] = ex / (ex + ey + 1e-6)
    f["lap_var"] = float(min(cv2.Laplacian(g, cv2.CV_32F).var(), 3000) / 3000)
    # top-right barcode: dense vertical stripes
    tr = g[: H // 8, W // 2:]
    f["barcode"] = float(np.abs(np.diff(tr.astype(np.float32), axis=1)).mean() / 255)
    f["v_mean"] = float(v.mean() / 255)
    return f


def field_features(c):
    """colour / texture descriptors for crop-state classification."""
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    f = {}
    exg = 2 * c[..., 1].astype(np.float32) - c[..., 0] - c[..., 2]
    g_ = c[..., 1].astype(np.float32); r_ = c[..., 2].astype(np.float32); b_ = c[..., 0].astype(np.float32)
    green = (exg > 20)
    yel = (hue >= 8) & (hue < 35) & (s > 50) & (v > 70)
    brown = (hue >= 5) & (hue < 25) & (s > 40) & (v < 140) & (v > 40)
    soil = (hue >= 5) & (hue < 25) & (s < 110) & (v > 110) & ~green
    sky = ((hue > 90) & (hue < 130) & (v > 130)) | ((s < 35) & (v > 195))
    top = slice(0, H // 3)
    f["green"] = float(green.mean()); f["yellow"] = float(yel.mean()); f["brown"] = float(brown.mean())
    f["soil"] = float(soil.mean()); f["sky"] = float(sky.mean()); f["sky_top"] = float(sky[top].mean())
    f["dark"] = float((v < 60).mean()); f["v_mean"] = float(v.mean() / 255); f["s_mean"] = float(s.mean() / 255)
    f["exg_mean"] = float(exg.mean() / 255)
    # greenness in the lower 2/3 (ground/crop area) excluding sky
    low = slice(H // 3, H)
    f["green_low"] = float(green[low].mean()); f["yellow_low"] = float(yel[low].mean()); f["soil_low"] = float(soil[low].mean())
    f["green_frac_of_veg"] = float(green.sum() / (green.sum() + yel.sum() + brown.sum() + 1))
    for i in range(3):
        f[f"green_row{i}"] = float(green[i * H // 3:(i + 1) * H // 3].mean())
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    f["lap"] = float(min(cv2.Laplacian(g, cv2.CV_32F).var(), 3000) / 3000)
    ed = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 50, 120)
    f["edge"] = float((ed > 0).mean())
    # texture regularity: rows/col anisotropy (crop rows)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)).mean(); gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
    f["aniso"] = float(gx / (gx + gy + 1e-6))
    # 4x3x3 hsv histogram
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [6, 3, 3], [0, 180, 0, 256, 0, 256]).ravel()
    hist = hist / hist.sum()
    for i, x in enumerate(hist):
        f[f"h{i}"] = float(x)
    return f


def orient_features(c):
    """Orientation-aware features (row / column profiles of brightness, greenness, texture, saturation, blue-red,
    plus a 4x6 grid). c: BGR content of any size (squashed to 48x64)."""
    c = cv2.resize(c, (48, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV).astype(np.float32)
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    exg = 2 * c[..., 1].astype(np.float32) - c[..., 0] - c[..., 2]
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)); gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    chans = [hsv[..., 2] / 255, hsv[..., 1] / 255, exg / 255, gx / 255, gy / 255, (c[..., 0].astype(np.float32) - c[..., 2]) / 255]
    out = []
    for ch in chans:
        out.extend(cv2.resize(ch, (4, 6), interpolation=cv2.INTER_AREA).ravel().tolist())
        out.extend(cv2.resize(ch.mean(axis=1, keepdims=True), (1, 16), interpolation=cv2.INTER_AREA).ravel().tolist())
        out.extend(cv2.resize(ch.mean(axis=0, keepdims=True), (12, 1), interpolation=cv2.INTER_AREA).ravel().tolist())
    tex = gx + gy
    out.append(float(tex[:21].mean() - tex[43:].mean())); out.append(float(tex[:, :16].mean() - tex[:, 32:].mean()))
    return np.array(out, np.float32)


def phash(c, size=8):
    """64-bit DCT perceptual hash of the content area (BGR) -> np.uint8 bits (size*size)."""
    g = cv2.cvtColor(cv2.resize(c, (32, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    d = cv2.dct(g)[:size, :size]
    med = np.median(d.ravel()[1:])
    return (d.ravel() > med).astype(np.uint8)


def small_gray(c, n=48):
    return cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)


def ssim(a, b):
    """global-window SSIM on small grayscale arrays (8x8 blocks, mean)."""
    C1, C2 = 6.5025, 58.5225
    k = (7, 7)
    mu1, mu2 = cv2.blur(a, k), cv2.blur(b, k)
    s11 = cv2.blur(a * a, k) - mu1 ** 2; s22 = cv2.blur(b * b, k) - mu2 ** 2; s12 = cv2.blur(a * b, k) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


def water_features(c):
    """Cheap flood cues on the content image (BGR, HxW=232x160): smooth, non-green, sky-coloured / grey-brown areas in the
    lower part of the frame (standing water reflects sky or shows muddy brown)."""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    mu = cv2.blur(g, (5, 5)); sd = np.sqrt(np.maximum(cv2.blur(g * g, (5, 5)) - mu * mu, 0))
    exg = 2 * c[..., 1].astype(np.float32) - c[..., 0] - c[..., 2]
    smooth = (sd < 5) & (exg < 12)
    lower = np.zeros_like(smooth); lower[int(H * 0.30):] = True
    skyc = ((hue > 85) & (hue < 135) & (s > 25)) | ((s < 40) & (v > 150))
    muddy = (hue >= 5) & (hue < 28) & (s > 30) & (s < 140) & (v > 50) & (v < 190)
    dark = (v < 90)
    f = {}
    f["w_smooth_low"] = float((smooth & lower).sum() / lower.sum())
    f["w_sky_low"] = float((smooth & lower & skyc).sum() / lower.sum())
    f["w_muddy_low"] = float((smooth & lower & muddy).sum() / lower.sum())
    f["w_dark_low"] = float((smooth & lower & dark).sum() / lower.sum())
    f["w_smooth_rows"] = float(((smooth & lower).mean(axis=1) > 0.5).sum() / H)
    f["sd_low"] = float(sd[lower].mean() / 255)
    return f
