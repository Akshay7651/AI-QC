"""Match locally downloaded PDFs/photos (a ZIP or folder) to rows by docket id.

A file belongs to a docket if the docket id appears as a token in its relative path,
so all of these work:  <docket>.pdf   <docket>_1.jpg   <docket>/photo2.jpg   pdfs/<docket>-form.pdf
"""
import re
import zipfile
from pathlib import Path

PDF_EXT = {".pdf"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _materialize(src: str, cache_root="cache/local") -> Path:
    p = Path(src)
    if p.is_dir():
        return p
    if p.suffix.lower() != ".zip" or not p.exists():
        raise FileNotFoundError(f"--local-media must be a .zip or a folder: {src}")
    dest = Path(cache_root) / p.stem
    marker = dest / ".extracted"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p) as z:
            z.extractall(dest)  # zipfile strips absolute paths / '..' components
        for inner in list(dest.rglob("*.zip")):  # one level of nested zips
            with zipfile.ZipFile(inner) as z:
                z.extractall(inner.with_suffix(""))
        marker.touch()
    return dest


def index(src: str, dockets) -> dict:
    """docket_id -> {"pdfs": [Path], "images": [Path]}"""
    root = _materialize(src)
    wanted = {str(d).strip(): str(d).strip() for d in dockets if d}
    lower = {k.lower(): k for k in wanted}
    out = {}
    for f in sorted(root.rglob("*")):
        ext = f.suffix.lower()
        if not f.is_file() or (ext not in PDF_EXT and ext not in IMG_EXT):
            continue
        rel = f.relative_to(root)
        tokens = re.split(r"[^A-Za-z0-9]+", str(rel.with_suffix("")))
        hit = next((lower[t.lower()] for t in reversed(tokens) if t.lower() in lower), None)
        if hit is None:
            hit = next((lower[s.lower()] for s in (f.stem, f.parent.name) if s.lower() in lower), None)
        if hit is not None:
            e = out.setdefault(hit, {"pdfs": [], "images": []})
            e["pdfs" if ext in PDF_EXT else "images"].append(f)
    return out
