"""Match locally downloaded PDFs/photos (a ZIP or folder) to rows by docket id.

A file belongs to a row if, in its relative path, it contains either
  - the docket id as a token:  <docket>.pdf   <docket>_1.jpg   <docket>/photo2.jpg   pdfs/<docket>-form.pdf
  - or the mediaID GUID from that row's Signed_Copy_URL / Media URLs (what the PMFBY download
    endpoint names files):  06DAD6A0-AA87-4006-9717-B4CAA9805548.pdf
"""
import hashlib
import re
import zipfile

import pandas as pd
from pathlib import Path

PDF_EXT = {".pdf"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _materialize(src: str, cache_root="cache/local") -> Path:
    p = Path(src)
    if p.is_dir():
        return p
    if p.suffix.lower() != ".zip" or not p.exists():
        raise FileNotFoundError(f"--local-media must be a .zip or a folder: {src}")
    st = p.stat()
    tag = hashlib.sha1(f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:8]
    dest = Path(cache_root) / f"{p.stem}_{tag}"  # unique per zip content, so same-named/updated zips don't collide
    marker = dest / ".extracted"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)  # zipfile strips absolute paths / '..' components
        for inner in list(dest.rglob("*.zip")):  # one level of nested zips
            try:
                with zipfile.ZipFile(inner) as z:
                    z.extractall(inner.with_suffix(""))
            except zipfile.BadZipFile:
                pass
        marker.touch()
    return dest


GUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def _roots(src: str, cache_root="cache/local") -> list:
    """Folders to scan. A single .zip -> its extracted folder. A folder -> the folder itself PLUS, for every .zip inside it
    (e.g. one ZIP per docket: <docket>.zip containing form/ and media/), an extracted copy under cache/ whose folder is
    named after the zip, so the docket id stays in the relative path."""
    p = Path(src)
    if not p.is_dir():
        return [_materialize(src, cache_root)]
    roots = [p]
    zips = [z for z in p.rglob("*.zip") if z.is_file()]
    if zips:
        tag = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:8]
        base = Path(cache_root) / f"dir_{tag}"
        for z in zips:
            dest = base / z.stem
            marker = dest / ".extracted"
            if not marker.exists():
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    with zipfile.ZipFile(z) as zf:
                        zf.extractall(dest)
                    marker.touch()
                except zipfile.BadZipFile:
                    continue
        roots.append(base)
    return roots


def index(src: str, df) -> dict:
    """docket_id -> {"pdfs": [Path], "images": [Path]}"""
    out = {}
    cache_abs = Path("cache/local").resolve()
    for root in _roots(src):
        _scan(root, df, out, skip=None if cache_abs in root.resolve().parents or root.resolve() == cache_abs or root.name.startswith("dir_") else cache_abs)
    return out


def _scan(root: Path, df, out: dict, skip=None):
    wanted = {str(d).strip() for d in df["docket_id"] if pd.notna(d) and str(d).strip()}
    lower = {k.lower(): k for k in wanted}
    guid_to_docket = {}
    for d, u1, u2 in zip(df["docket_id"], df["pdf_url"], df["media_urls"]):
        if pd.isna(d):
            continue
        u1, u2 = ("" if pd.isna(u) else u for u in (u1, u2))
        for g in GUID.findall(f"{u1 or ''} {u2 or ''}"):
            guid_to_docket[g.lower()] = str(d).strip()
    for f in sorted(root.rglob("*")):
        ext = f.suffix.lower()
        if not f.is_file() or (ext not in PDF_EXT and ext not in IMG_EXT):
            continue
        if skip is not None and skip in f.resolve().parents:   # our own extraction cache inside the scanned folder
            continue
        rel = f.relative_to(root)
        tokens = re.split(r"[^A-Za-z0-9]+", str(rel.with_suffix("")))
        hit = next((guid_to_docket[g.lower()] for g in GUID.findall(str(rel)) if g.lower() in guid_to_docket), None)
        if hit is None:
            hit = next((lower[t.lower()] for t in reversed(tokens) if t.lower() in lower), None)
        if hit is None:
            hit = next((lower[s.lower()] for s in (f.stem, f.parent.name) if s.lower() in lower), None)
        if hit is not None:
            e = out.setdefault(hit, {"pdfs": [], "images": []})
            e["pdfs" if ext in PDF_EXT else "images"].append(f)

