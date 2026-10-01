"""Fast offline evidence-photo analyst (no API, no GPU).  analyse(paths, row, form_image=None) -> dict.

Per photo (cached on disk by path+size+mtime):
  * burned-in GPS-camera stamp (Latitude/Longitude/Date) read with Tesseract on a thresholded bottom-left crop,
    all photos of a row stacked into ONE Tesseract call (photo_stamp.py); EXIF fallback
  * "is this a photo of the paper form / a document?"  image statistics (photo_feats.form_features) ->
    RF+ET+LR ensemble (photo_models/form.pkl); ambiguous ones get a Devanagari keyword OCR and, if the row's form
    jpg is given, an ORB feature match against it
  * orientation (sideways photos): stamp OCR rotation, form-line rule, and an ExtraTrees orientation model
    trained on synthetically rotated photos (photo_models/orient.pkl); photos are auto-rotated upright
  * person present (YuNet face detector, photo_models/yunet.onnx)
  * crop visible? (RF on colour/texture, photo_models/crop_visible.pkl) + rule-based field state / loss hint
  * perceptual hash + SSIM for near-duplicates
Row level: GPS distance stamp<->app, photo date vs loss date / intimation date, duplicates, remarks.
"""
import os
import pickle
import joblib
import re
import threading
import warnings
from datetime import datetime

import cv2
import numpy as np

import config as C
import photo_feats as PF
import photo_stamp as PS
from common import haversine_m, num, parse_dates

os.environ.setdefault("OMP_THREAD_LIMIT", "1")
warnings.filterwarnings("ignore")
cv2.setNumThreads(1)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "photo_models")
_CACHE_DIR = os.path.join(_HERE, getattr(C, "PHOTO_CACHE_DIR", "cache/photos"))
_LOCK = threading.Lock()
_M = {}
_MV = None

FIELD_LABELS = ("no crop", "cut & spread", "crop mismatch", "standing crop")
DUP_HD = 12            # pHash Hamming distance (of 64) for a near-duplicate
DUP_SSIM = 0.70        # ... or SSIM (48x48 grey) above this together with hd <= 20
FORM_AMBIG = (0.2, 0.8)
ROT_TH = 0.8
FORM_SIDEWAYS_EDGE_XY = 0.56
KEYWORDS = ("प्रारूप", "बीमा", "हस्ताक्षर", "प्रधानमंत्री", "फसल", "कृषक", "सर्वे", "नुकसान", "क्षति", "खसरा", "कम्पनी", "अधिकारी", "PMFBY")


# ---------------------------------------------------------------- models
def _iter_models(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_models(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_models(v)
    else:
        yield obj
        if hasattr(obj, "steps"):
            for _, st in obj.steps:
                yield st


def _load(name):
    with _LOCK:
        if name not in _M:
            p = os.path.join(_MODELS, name)
            if name.endswith(".pkl"):
                _M[name] = joblib.load(p) if os.path.exists(p) else None
                for m in _iter_models(_M[name]):
                    if hasattr(m, "n_jobs"):
                        m.n_jobs = 1
            elif name.endswith(".onnx"):
                try:
                    _M[name] = cv2.FaceDetectorYN.create(p, "", (240, 350), 0.5, 0.3, 50) if os.path.exists(p) else None
                except Exception:
                    _M[name] = None
        return _M[name]


def _form_prob(featdf_rows):
    m = _load("form.pkl")
    if m is None:
        return np.array([0.5] * len(featdf_rows))
    X = np.array([[f[c] for c in m["cols"]] for f in featdf_rows])
    return np.mean([mm.predict_proba(X)[:, 1] for mm in m["models"]], axis=0)


def _orient_probs(contents):
    m = _load("orient.pkl")
    if m is None or not contents:
        return np.tile([1.0, 0, 0, 0], (len(contents), 1))
    X = np.array([PF.orient_features(c) for c in contents])
    return m.predict_proba(X)          # classes 0, 90, 180, 270 (rotation that was applied to an upright photo)


_ROTBACK = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def _rotate_back(img, k):
    return cv2.rotate(img, _ROTBACK[k]) if k in _ROTBACK else img


_BACKBONE = "image_classification_mobilenetv2_2022apr.onnx"


def backbone_available():
    return os.path.exists(os.path.join(_MODELS, _BACKBONE))


def _backbone_logits(small_bgr):
    """MobileNetV2 (ImageNet, frozen) 1000-d logits used as an embedding; None if the ONNX file is missing
    (download it with tools/fetch_models.py)."""
    with _LOCK:
        if _BACKBONE not in _M:
            try:
                _M[_BACKBONE] = cv2.dnn.readNetFromONNX(os.path.join(_MODELS, _BACKBONE)) if backbone_available() else None
            except Exception:
                _M[_BACKBONE] = None
        net = _M[_BACKBONE]
        if net is None:
            return None
        x = cv2.resize(small_bgr, (224, 224), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        net.setInput(x.transpose(2, 0, 1)[None].copy())
        return net.forward()[0].astype(np.float32)


def _faces(c):
    fd = _load("yunet.onnx")
    if fd is None:
        return 0, 0.0
    try:
        s = cv2.resize(c, (240, 350))
        fd.setInputSize((240, 350))
        _, f = fd.detect(s)
        return (0, 0.0) if f is None else (len(f), float(f[:, -1].max()))
    except Exception:
        return 0, 0.0


# ---------------------------------------------------------------- exif
def _exif(path):
    out = {"date": None, "lat": None, "lng": None}
    try:
        from PIL import Image
        ex = Image.open(path).getexif()
        dt = ex.get(306) or ex.get_ifd(0x8769).get(36867)
        if dt:
            out["date"] = datetime.strptime(str(dt)[:10], "%Y:%m:%d").strftime("%d%m%Y")
        gps = ex.get_ifd(0x8825)
        if gps and 2 in gps and 4 in gps:
            conv = lambda v, ref: (float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600) * (-1 if ref in "SW" else 1)
            out["lat"], out["lng"] = conv(gps[2], gps[1]), conv(gps[4], gps[3])
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- field state
def _field_state(ff, p_crop):
    """Map colour statistics + crop-visible probability to the four allowed labels plus a condition note.
    'standing crop' vs 'crop mismatch' is learned (weak: few negatives); 'no crop' and 'cut & spread' are rules
    that could not be validated (no examples in the labelled set)."""
    veg = ff["green"] + ff["yellow"] + ff["brown"]
    note = ""
    if ff["sky"] > 0.35 and ff["green_low"] < 0.15 and veg < 0.2:
        note = "waterlogged?"
    if ff["aniso"] < 0.42 and ff["yellow_low"] + ff["brown"] > 0.35 and ff["green"] < 0.25:
        note = "lodged/cut?"
    if veg < 0.06 and ff["soil"] + ff["dark"] > 0.25:
        label = "no crop"
    elif note == "lodged/cut?" and ff["green"] < 0.12:
        label = "cut & spread"
    elif p_crop < 0.4:
        label = "crop mismatch"
    else:
        label = "standing crop"
    if not note and ff["yellow"] + ff["brown"] > 0.45 and ff["green"] < 0.15:
        note = "dry"
    # crude visible-damage index: share of vegetation that is not green, steps of 5 (rough, see report)
    dmg = 1.0 - ff["green_frac_of_veg"]
    loss = int(round(min(max(dmg, 0.0), 1.0) * 100 / 5) * 5)
    return label, note, loss


# ---------------------------------------------------------------- per photo
def _models_version():
    """Changes whenever a model file is retrained, so cached predictions never go stale."""
    try:
        t = 0
        for f in ("form.pkl", "orient.pkl", "crop_visible.pkl", "yunet.onnx", "heads.pkl", _BACKBONE):
            p = os.path.join(_MODELS, f)
            t += int(os.stat(p).st_mtime) if os.path.exists(p) else 7
        return "m%x" % (t % 0xFFFFFF)
    except OSError:
        return "m0"


def _key(path):
    global _MV
    if _MV is None:
        _MV = _models_version()
    try:
        st = os.stat(path)
        return f"{os.path.basename(path)}.{st.st_size}.{int(st.st_mtime)}.{_MV}.pkl"
    except OSError:
        return None


def _cache_get(path):
    k = _key(path)
    if not k:
        return None
    p = os.path.join(_CACHE_DIR, k)
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_put(path, d):
    k = _key(path)
    if not k:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = os.path.join(_CACHE_DIR, f".{os.getpid()}.{threading.get_ident()}.{k}")
        with open(tmp, "wb") as f:
            pickle.dump(d, f)
        os.replace(tmp, os.path.join(_CACHE_DIR, k))
    except Exception:
        pass


def _ocr_keywords(bgr):
    """Devanagari OCR on a downscaled copy; returns (n_devanagari_chars, n_keywords). ~1-3 s: only for ambiguous photos."""
    try:
        import pytesseract
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[: int(bgr.shape[0] * 0.75)]
        g = cv2.resize(g, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        t = pytesseract.image_to_string(g, lang="hin", config="--psm 11", timeout=8)
        return len(re.findall(r"[ऀ-ॿ]", t)), sum(k.lower() in t.lower() for k in KEYWORDS)
    except Exception:
        return None, None


_FORM_ORB = {}


def _orb_match(photo_gray, form_path):
    """Number of RANSAC inliers between an evidence photo and the row's form jpg (same paper => many)."""
    try:
        orb = cv2.ORB_create(700)
        if form_path not in _FORM_ORB:
            fg = cv2.imread(form_path, 0)
            fg = cv2.resize(fg, None, fx=800.0 / max(fg.shape), fy=800.0 / max(fg.shape), interpolation=cv2.INTER_AREA)
            kp, de = orb.detectAndCompute(fg, None)
            if len(_FORM_ORB) > 64:
                _FORM_ORB.clear()
            _FORM_ORB[form_path] = (kp, de)
        kf, df = _FORM_ORB[form_path]
        pg = cv2.resize(photo_gray, None, fx=800.0 / max(photo_gray.shape), fy=800.0 / max(photo_gray.shape), interpolation=cv2.INTER_AREA)
        kp, de = orb.detectAndCompute(pg, None)
        if de is None or df is None or len(kp) < 10 or len(kf) < 10:
            return 0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        good = [m for m, n in (p for p in bf.knnMatch(de, df, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            return len(good) // 2
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kf[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        return int(mask.sum()) if mask is not None else 0
    except Exception:
        return 0


def _analyse_new(paths):
    """Decode + analyse photos that are not cached. Returns list of per-photo dicts (app-independent)."""
    imgs = [cv2.imread(p) for p in paths]
    out = [None] * len(paths)
    ok = [i for i, im in enumerate(imgs) if im is not None and im.size]
    if not ok:
        return out
    grays = [cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY) for i in ok]
    stamps = PS.read_stamps(grays)
    for j, i in enumerate(ok):
        st = stamps[j]
        img = imgs[i]
        d = {"path": paths[i], "shape": img.shape[:2], "stamp": st}
        rot = st.get("rot", 0) if st.get("stamp_ok") else 0
        if rot:
            # stamp text was sideways: the whole frame is rotated; bring it upright
            img = cv2.rotate(img, {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 270: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}[rot])
        d["rot_stamp"] = rot
        c = PF.content(img)
        d["_img_small"] = cv2.resize(img[: int(img.shape[0] * PF.STAMP_FRAC)], (360, 525), interpolation=cv2.INTER_AREA)
        d["form_f"] = PF.form_features(c)
        d["exif"] = _exif(paths[i]) if not st.get("stamp_ok") or st.get("date") is None else {"date": None, "lat": None, "lng": None}
        out[i] = d
    # batch model calls (form probability first; orientation / crop models only for photos that are not forms)
    idx = list(ok)
    pf = _form_prob([out[i]["form_f"] for i in idx])
    for j, i in enumerate(idx):
        out[i]["p_form"] = float(pf[j])
    fld = [i for i in idx if out[i]["p_form"] < 0.5]
    po = _orient_probs([out[i]["_img_small"] for i in fld])
    for j, i in enumerate(fld):
        out[i]["orient_p"] = po[j].tolist()
    for i in idx:
        d = out[i]
        d.setdefault("orient_p", [1.0, 0.0, 0.0, 0.0])
        k = 0
        pr = np.array(d["orient_p"])
        if d["p_form"] < 0.5 and 1 - pr[0] > ROT_TH:
            k = [90, 180, 270][int(np.argmax(pr[1:]))]      # sideways forms are not detected (striped backgrounds fool line statistics)
        d["rot_content"] = k
        d["rotated"] = bool(d["rot_stamp"] or k)
        if k:
            im2 = _rotate_back(d["_img_small"], k)
            c = cv2.resize(im2, (PF.W, PF.H), interpolation=cv2.INTER_AREA)
            d["form_f"] = PF.form_features(c)
            d["p_form"] = float(_form_prob([d["form_f"]])[0])
        else:
            c = cv2.resize(d["_img_small"], (PF.W, PF.H), interpolation=cv2.INTER_AREA)
        d["field_f"] = PF.field_features(c)
        d["hash"] = PF.phash(d["_img_small"])
        d["sg"] = PF.small_gray(d["_img_small"])
        d["nfaces"], d["face_conf"] = _faces(_rotate_back(d["_img_small"], k) if k else d["_img_small"])
        small_up = _rotate_back(d["_img_small"], k) if k else d["_img_small"]
        if d["p_form"] < 0.5:
            lg = _backbone_logits(small_up)
            d["logits"] = None if lg is None else lg.astype(np.float16)
            d["water"] = PF.water_features(c)
        d["quality"] = "dark" if d["field_f"]["v_mean"] < 0.18 else "blurry" if d["field_f"]["lap"] < 0.004 else "good"
    cv_ = _load("crop_visible.pkl")
    fl2 = [i for i in idx if out[i]["p_form"] < 0.5]
    for i in idx:
        out[i]["p_crop"] = 0.5
    if cv_ is not None and fl2:
        pc = cv_["model"].predict_proba(np.array([[out[i]["field_f"][c_] for c_ in cv_["cols"]] for i in fl2]))[:, 1]
        for j, i in enumerate(fl2):
            out[i]["p_crop"] = float(pc[j])
    for d in out:
        if d is not None:
            d.pop("_img_small", None)
    return out


def _small_upright(d):
    """Re-read a photo (only needed for the rare OCR / ORB checks) as an upright 360x525 content crop."""
    img = cv2.imread(d["path"])
    if d.get("rot_stamp"):
        img = cv2.rotate(img, {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 270: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}[d["rot_stamp"]])
    img = cv2.resize(img[: int(img.shape[0] * PF.STAMP_FRAC)], (360, 525), interpolation=cv2.INTER_AREA)
    return _rotate_back(img, d.get("rot_content", 0))


def _get_photos(paths):
    res = [None] * len(paths)
    todo = []
    for i, p in enumerate(paths):
        c = _cache_get(p)
        if c is not None:
            res[i] = c
        else:
            todo.append(i)
    if todo:
        new = _analyse_new([paths[i] for i in todo])
        for i, d in zip(todo, new):
            res[i] = d
            if d is not None:
                _cache_put(paths[i], d)
    return res


# ---------------------------------------------------------------- helpers
def _fmt_dist(m):
    return f"{m / 1000:.1f} km" if m >= 1000 else f"{m:.0f} m"


def _date_obj(s):
    return datetime.strptime(s, "%d%m%Y")


def _row_date(row, key):
    import pandas as pd
    try:
        v = parse_dates([row.get(key)])[0]
        return v if pd.notna(v) else None
    except Exception:
        return None


def analyse(paths, row: dict, form_image=None) -> dict:
    import pandas as pd
    paths = [str(p) for p in list(paths)[: C.MAX_PHOTOS_PER_ROW]]
    photos = [(p, d) for p, d in zip(paths, _get_photos(paths)) if d is not None]
    if not photos:
        return {"photo_status": "Unavailable", "photo_error": "no readable photos", "remarks": [], "n_photos": 0}
    paths = [p for p, _ in photos]
    P = [d for _, d in photos]
    n = len(P)
    lat, lng = num(row.get("latitude")), num(row.get("longitude"))
    remarks, flags = [], []

    # ---- 1. form / document photos -------------------------------------------------------------
    is_form, orbs, ocr_info = [], [None] * n, [None] * n
    n_ocr = 0
    for i, d in enumerate(P):
        p = d["p_form"]
        f = p >= 0.5
        if form_image and os.path.exists(str(form_image)) and 0.1 < p < 0.97:
            orbs[i] = _orb_match(cv2.cvtColor(_small_upright(d), cv2.COLOR_BGR2GRAY), str(form_image))
            if orbs[i] >= 25:
                f = True
        if FORM_AMBIG[0] < p < FORM_AMBIG[1] and not (orbs[i] is not None and orbs[i] >= 25) and n_ocr < 2:
            n_ocr += 1
            nd, nk = _ocr_keywords(_small_upright(d))
            ocr_info[i] = (nd, nk)
            if nd is not None:
                if nk >= 2 or nd >= 250:
                    f = True
                elif nd < 60 and nk == 0:
                    f = False
        is_form.append(bool(f))
    n_form = int(sum(is_form))
    if n_form == n:
        remarks.append(f"All {n} uploaded photo{'s are' if n > 1 else ' is'} of the paper form - no field photograph uploaded")
        flags.append("No field photograph (form photographed instead)")
    elif n_form:
        ids = ", ".join(str(i + 1) for i, f in enumerate(is_form) if f)
        remarks.append(f"Photo {ids} {'is' if n_form == 1 else 'are'} of the paper form, not the field")
        flags.append("Some photos are of the paper form")
    for i, o in enumerate(orbs):
        if o is not None and o >= 25 and is_form[i]:
            remarks.append(f"Photo {i + 1} shows the same paper form as the uploaded form image")
            break

    # ---- 2. duplicates -------------------------------------------------------------------------
    dup_of = [None] * n
    for j in range(1, n):
        for i in range(j):
            hd = int((P[i]["hash"] != P[j]["hash"]).sum())
            ss = PF.ssim(P[i]["sg"], P[j]["sg"])
            if hd <= DUP_HD or (ss >= DUP_SSIM and hd <= 20):
                dup_of[j] = i
                break
    n_dup = int(sum(x is not None for x in dup_of))
    for j, i in enumerate(dup_of):
        if i is not None:
            remarks.append(f"Photo {j + 1} is a duplicate / near-duplicate of photo {i + 1}")
    if n_dup:
        flags.append("Duplicate photos")

    # ---- 3. stamp / EXIF GPS + date -----------------------------------------------------------
    ref = (lat, lng)
    stamps, dists, unconf = [], [], []
    photo_dates = []
    for i, d in enumerate(P):
        st = dict(d["stamp"])
        ex = d["exif"]
        if not st.get("stamp_ok") and ex.get("lat") is not None:
            st.update(lat=ex["lat"], lng=ex["lng"], stamp_ok=True, src="exif")
        if not st.get("date") and ex.get("date"):
            st["date"] = ex["date"]
        gray_for_verify = None
        if st.get("stamp_ok") and lat is not None and lng is not None and haversine_m(lat, lng, st["lat"], st["lng"]) > C.GPS_PHOTO_MAX_DISTANCE_M and st.get("src") != "exif":
            gray_for_verify = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2GRAY)
            if d.get("rot_stamp"):
                gray_for_verify = cv2.rotate(gray_for_verify, {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 270: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180}[d["rot_stamp"]])
            st2, conf = PS.verify(gray_for_verify, st, ref, C.GPS_PHOTO_MAX_DISTANCE_M)
            unconf.append(conf is not True)
            st = st2
        else:
            unconf.append(False)
        stamps.append(st)
        dist = haversine_m(lat, lng, st["lat"], st["lng"]) if st.get("stamp_ok") and lat is not None and lng is not None else None
        dists.append(dist)
        photo_dates.append(st.get("date"))
    ok_st = [s for s in stamps if s.get("stamp_ok")]
    for i, s in enumerate(stamps):
        if not s.get("stamp_ok"):
            remarks.append(f"Photo {i + 1}: burned-in GPS stamp could not be read")
    far = [(i, dd) for i, dd in enumerate(dists) if dd is not None and dd > C.GPS_PHOTO_MAX_DISTANCE_M]
    for i, dd in far:
        remarks.append(f"Photo {i + 1} GPS (stamp) is {_fmt_dist(dd)} from app location" + (" (OCR not confirmed)" if unconf[i] else ""))
    if far:
        flags.append("GPS mismatch > %dm" % C.GPS_PHOTO_MAX_DISTANCE_M)
    if len(ok_st) >= 2:
        pts = [(s["lat"], s["lng"]) for s in ok_st]
        spread = max(haversine_m(a[0], a[1], b[0], b[1]) for a in pts for b in pts)
        if spread > 200:
            remarks.append(f"Photos were taken {_fmt_dist(spread)} apart from each other")
    stamp_lat = float(np.median([s["lat"] for s in ok_st])) if ok_st else None
    stamp_lng = float(np.median([s["lng"] for s in ok_st])) if ok_st else None
    vd = [dd for dd in dists if dd is not None]
    stamp_dist = float(max(vd)) if vd else None

    # ---- 4. dates -------------------------------------------------------------------------------
    dates = [x for x in photo_dates if x]
    photo_date = max(set(dates), key=dates.count) if dates else None
    loss_d, intim_d = _row_date(row, "survey_start_date"), _row_date(row, "survey_end_date")
    lag_intim = lag_loss = None
    date_bad = False
    if photo_date:
        pdt = pd.Timestamp(_date_obj(photo_date))
        if intim_d is not None:
            lag_intim = int((pdt - intim_d.normalize()).days)
        if loss_d is not None:
            lag_loss = int((pdt - loss_d.normalize()).days)
        if loss_d is not None and pdt < loss_d.normalize():
            remarks.append(f"Photo taken {-lag_loss} day{'s' if lag_loss != -1 else ''} before the loss date")
            date_bad = True
        if intim_d is not None and lag_intim > 30:
            remarks.append(f"Photo taken {lag_intim} days after intimation")
            date_bad = True
        elif intim_d is not None and lag_intim > 0 and not date_bad:
            remarks.append(f"Photo taken {lag_intim} day{'s' if lag_intim != 1 else ''} after intimation")
        if len(set(dates)) > 1:
            dd_ = sorted(_date_obj(x) for x in set(dates))
            if (dd_[-1] - dd_[0]).days > 7:
                remarks.append(f"Photo dates differ by {(dd_[-1] - dd_[0]).days} days")
        if date_bad or (lag_intim is not None and lag_loss is not None and photo_date_suspicious_safe(pdt, row)):
            flags.append("Photo date outside survey period")
    # ---- 5. rotation -------------------------------------------------------------------------------
    rot_ids = [i + 1 for i, d in enumerate(P) if d["rotated"]]
    if rot_ids:
        remarks.append(f"Photo {', '.join(map(str, rot_ids))} {'is' if len(rot_ids) == 1 else 'are'} rotated sideways (auto-rotated for analysis)")

    # ---- 6. field state: crop / flood / damage / scene (two independent models, see _field_outputs) ----------
    field_idx = [i for i in range(n) if not is_form[i]]
    fo = _field_outputs([P[i] for i in field_idx], row, n_form, n)
    remarks.extend(fo.pop("remarks"))
    field, note, est = fo.pop("field_photo"), fo.pop("field_note"), fo.pop("photo_loss")
    if fo.get("photo_conf") == "low" and field_idx:
        flags.append("Field-photo interpretation low confidence - manual check")
    face_idx = [i for i in range(n) if P[i]["nfaces"] > 0]
    farmer = bool(face_idx)
    if farmer:
        remarks.append(f"A person is visible in photo {', '.join(str(i + 1) for i in face_idx)}")
    loss = num(row.get("crop_loss_pct"))
    qual = max(set(d["quality"] for d in P), key=[d["quality"] for d in P].count)
    res = {
        "photo_status": "OK", "engine": "local-v3", "n_photos": n,
        "field_photo": field, "field_note": note, "farmer_photo": farmer, "person_each": [bool(d["nfaces"] > 0) for d in P],
        "photo_loss": est, "photo_loss_note": "rough colour-based hint only; not correlated with app loss in tests" if est is not None else "",
        "photo_date": photo_date, "photo_quality": qual, "photo_flags": flags,
        "photo_is_form": bool(n_form == n), "n_form_photos": n_form, "photo_is_form_each": is_form,
        "n_duplicates": n_dup, "duplicate_of": [None if x is None else x + 1 for x in dup_of],
        "stamp_lat": stamp_lat, "stamp_lng": stamp_lng, "stamp_dist_m": stamp_dist,
        "stamp_dist_each_m": dists, "stamp_unconfirmed": unconf,
        "stamp_date": photo_date, "stamp_read": [bool(s.get("stamp_ok")) for s in stamps],
        "photo_lag_days_after_intimation": lag_intim, "photo_days_after_loss": lag_loss,
        "rotated": bool(rot_ids), "remarks": remarks,
    }
    res.update(fo)
    return res


_CROP_SHORT = {"cotton": "cotton", "kapas": "cotton", "pearl millet": "pearl millet", "bajra": "pearl millet", "green gram": "green gram",
               "moong": "green gram", "paddy": "paddy", "dhan": "paddy", "rice": "paddy"}


def _declared_crop(row):
    v = str(row.get("crop_name") or "").lower()
    for k, short in _CROP_SHORT.items():
        if k in v:
            return short
    return None


def _head_probs(heads, task, Xb, Xc):
    """Return (probs_bb|None, probs_cl, classes) for a task; bb is None if the backbone is unavailable."""
    h = heads[task]
    pcl = h["cl"].predict_proba(Xc)
    pbb = h["bb"].predict_proba(Xb) if Xb is not None else None
    return pbb, pcl, list(h["classes"])


def _field_outputs(Pf, row, n_form, n):
    """Row-level crop / flood / damage / scene outputs from the non-form photos.  Two models vote per task: a logistic head on
    frozen MobileNetV2 embeddings and a random forest on colour/texture/water features; probabilities are averaged, and when the
    two argmaxes differ the row is marked photo_agree=False / low confidence."""
    out = {"crop_present": "no", "crop_present_conf": None, "crop_seen": "unknown", "crop_seen_conf": None, "crop_matches_declared": None,
           "flooded": "no", "water_frac": None, "damage_state": "", "damage_visible": False, "scene_type": "paper form" if n_form == n else "unknown",
           "photo_agree": None, "photo_conf": "low", "remarks": [], "field_photo": "no crop", "field_note": "no field photograph", "photo_loss": None}
    Pf = [d for d in (Pf or []) if "water" in d and "field_f" in d]     # photos treated as the paper form carry no field features
    if not Pf:
        return out
    heads = _load("heads.pkl")
    rem = out["remarks"]
    declared = _declared_crop(row)
    Xc = np.array([[d["field_f"][c] for c in heads["cols_cl"][:len(d["field_f"])]] + [d["water"][c] for c in heads["cols_cl"][len(d["field_f"]):]] for d in Pf], np.float32) if heads else None
    have_bb = heads is not None and all(d.get("logits") is not None for d in Pf)
    Xb = np.array([d["logits"].astype(np.float32) for d in Pf]) if have_bb else None
    if heads is None:
        # no trained heads: fall back to the older colour model; low confidence
        labs = [_field_state(d["field_f"], d["p_crop"]) for d in Pf]
        from collections import Counter
        out.update(field_photo=Counter(l for l, _, _ in labs).most_common(1)[0][0], field_note=labs[0][1], photo_loss=labs[0][2],
                   scene_type="field", crop_present="yes" if labs[0][0] == "standing crop" else "no", photo_conf="low")
        return out

    def combine(task):
        pbb, pcl, classes = _head_probs(heads, task, Xb, Xc)
        p = pcl if pbb is None else (pbb + pcl) / 2
        agree = None if pbb is None else (pbb.argmax(1) == pcl.argmax(1))
        return p, classes, agree

    agrees = []
    # --- scene: person-only vs field
    p, cl, ag = combine("scene")
    pers = p[:, cl.index("person-only")] if "person-only" in cl else np.zeros(len(Pf))
    person_only = [bool(pers[i] >= 0.5 or (Pf[i]["nfaces"] > 0 and Pf[i]["face_conf"] > 0.85 and pers[i] >= 0.3)) for i in range(len(Pf))]
    if ag is not None:
        agrees.append(ag.all())
    fi = [i for i in range(len(Pf)) if not person_only[i]]
    if not fi:
        out.update(scene_type="person-only", photo_conf="high" if (ag is None or ag.all()) else "low", crop_present="no", field_photo="crop mismatch",
                   field_note="person only, no crop visible", photo_agree=None if ag is None else bool(ag.all()))
        rem.append("The field photo shows only a person, no crop or field is visible")
        return out
    Xb2 = None if Xb is None else Xb[fi]; Xc2 = Xc[fi]
    sub = lambda task: (lambda pb, pc, cs: (pb[fi] if pb is not None else None, pc[fi], cs))(*_head_probs(heads, task, Xb, Xc))
    def comb2(task):
        pbb, pcl, classes = sub(task)
        return (pcl if pbb is None else (pbb + pcl) / 2), classes, (None if pbb is None else (pbb.argmax(1) == pcl.argmax(1)))
    pc_, cs_c, ag_c = comb2("crop_present"); pf_, cs_f, ag_f = comb2("flooded"); pd_, cs_d, ag_d = comb2("damage"); pt_, cs_t, ag_t = comb2("crop_type")
    for a_ in (ag_c, ag_f, ag_d):
        if a_ is not None:
            agrees.append(bool(a_.all()))
    # crop present (mean over field photos)
    pcrop = float(pc_[:, cs_c.index(1)].mean()) if 1 in cs_c else 0.5
    pflood = float(pf_[:, cs_f.index(1)].mean()) if 1 in cs_f else 0.0
    dmean = pd_.mean(0); dstate = str(cs_d[int(dmean.argmax())]); dconf = float(dmean.max())
    tmean = pt_.mean(0); tstate = str(cs_t[int(tmean.argmax())]); tconf = float(tmean.max())
    wf = float(np.mean([min(1.0, Pf[i]["water"]["w_smooth_low"]) for i in fi]))
    no_crop_state = dstate in ("weeds-uncultivated", "bare soil", "harvested")
    crop_yes = (pcrop >= 0.5) and not (no_crop_state and dconf >= 0.5)
    out["crop_present"] = "yes" if crop_yes else "no"; out["crop_present_conf"] = round(max(pcrop, 1 - pcrop), 2)
    out["flooded"] = "yes" if (pflood >= 0.5 or dstate == "submerged") else "no"; out["water_frac"] = round(wf, 2)
    out["damage_state"] = dstate; out["damage_visible"] = bool(dstate not in ("healthy",))
    out["scene_type"] = "field"
    sky = float(np.mean([Pf[i]["field_f"]["sky"] for i in fi])); veg = float(np.mean([Pf[i]["field_f"]["green"] + Pf[i]["field_f"]["yellow"] + Pf[i]["field_f"]["brown"] for i in fi]))
    if any(Pf[i]["quality"] != "good" for i in fi) and all(Pf[i]["quality"] != "good" for i in fi):
        out["scene_type"] = "blurry-dark-irrelevant"
    elif sky > 0.5 and veg < 0.1:
        out["scene_type"] = "house-road-sky-other"      # rule only: no labelled examples
    if tconf >= 0.5:
        out["crop_seen"] = str(tstate); out["crop_seen_conf"] = round(tconf, 2)
        if declared:
            out["crop_matches_declared"] = bool(tstate == declared)
    else:
        out["crop_seen_conf"] = round(tconf, 2)
    out["photo_agree"] = bool(all(agrees)) if agrees else None
    low = (out["photo_agree"] is False) or dconf < 0.5 or out["crop_present_conf"] < 0.6
    out["photo_conf"] = "low" if (low or not have_bb) else "high"
    # legacy four-label field_photo
    out["field_photo"] = ("crop mismatch" if dstate == "weeds-uncultivated" else "no crop" if dstate in ("bare soil", "harvested") else
                          "cut & spread" if dstate == "cut & spread" else "crop mismatch" if not crop_yes else "standing crop")
    out["field_note"] = "; ".join(x for x in (("flooded" if out["flooded"] == "yes" else ""), dstate if dstate not in ("healthy", "") else "") if x)
    out["photo_loss"] = None
    # remarks
    dec = f" (declared: {declared})" if declared else ""
    if dstate == "weeds-uncultivated":
        rem.append(f"Photo shows weeds / uncultivated land, no insured crop visible{dec}")
    elif not crop_yes:
        rem.append(f"No crop visible in the field photo ({dstate or 'unclear'}){dec}")
    if out["flooded"] == "yes":
        rem.append("Field appears flooded / waterlogged (water visible)")
    if crop_yes and dstate not in ("healthy", "weeds-uncultivated"):
        rem.append(f"Visible crop condition: {dstate}")
    if out["crop_matches_declared"] is False:
        rem.append(f"Crop in photo looks like {out['crop_seen']}, declared crop is {declared}")
    if out["photo_agree"] is False:
        rem.append("The two photo models disagree - manual review")
    elif out["photo_conf"] == "low":
        rem.append("Field-photo interpretation is low confidence - manual check")
    return out


def photo_date_suspicious_safe(pdt, row):
    try:
        from common import photo_date_suspicious
        return bool(photo_date_suspicious(pdt, row))
    except Exception:
        return False
