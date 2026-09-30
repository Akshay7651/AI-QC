"""Load any CLAP export (xlsx/xls/csv/folder/URL/Drive link), detect headers, map to canonical columns."""
import io
import re
from pathlib import Path

import pandas as pd

from column_map import COLUMN_ALIASES, DEFAULT_ORDER, map_columns

NUMERIC = ["farm_area", "affected_area_pct", "crop_loss_pct", "total_damage_pct", "latitude", "longitude"]


def _download(url: str) -> bytes:
    import httpx
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([\w-]+)", url)
    if m:
        url = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    r = httpx.get(url, follow_redirects=True, timeout=120)
    r.raise_for_status()
    return r.content


def _read_raw(src: str) -> pd.DataFrame:
    """Read a single file with header=None so we can decide about headers ourselves."""
    if src.startswith(("http://", "https://")):
        data = _download(src)
        if data[:2] == b"PK":
            return pd.read_excel(io.BytesIO(data), header=None, dtype=str)
        try:
            return pd.read_excel(io.BytesIO(data), header=None, dtype=str)
        except Exception:
            return pd.read_csv(io.BytesIO(data), header=None, dtype=str)
    if src.lower().endswith(".csv"):
        return pd.read_csv(src, header=None, dtype=str, encoding_errors="replace")
    return pd.read_excel(src, header=None, dtype=str)


def has_header(first_row) -> bool:
    cells = [str(c).strip() for c in first_row if pd.notna(c) and str(c).strip()]
    if not cells:
        return False
    if len(map_columns(cells)) >= 3:
        return True
    if any(re.fullmatch(r"-?\d+(\.\d+)?", c) for c in cells):
        return False  # header cells are never plain numbers; data rows usually are
    first = cells[0]
    if re.fullmatch(r"\d{10,}", first) or first.lower().startswith(("http://", "https://")):
        return False
    return bool(re.search(r"[A-Za-z]", first))


def _to_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=DEFAULT_ORDER)
    if has_header(raw.iloc[0]):
        headers = [str(h).strip() if pd.notna(h) else "" for h in raw.iloc[0]]
        body = raw.iloc[1:].reset_index(drop=True)
        mapping = map_columns(headers)
        names = list(headers)
        for canon, idx in mapping.items():
            names[idx] = canon
    else:
        body = raw.reset_index(drop=True)
        names = [DEFAULT_ORDER[i] if i < len(DEFAULT_ORDER) else f"extra_{i}" for i in range(raw.shape[1])]
    # de-duplicate any repeated header names
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    body.columns = out
    return body


def load(src: str) -> pd.DataFrame:
    """Return a DataFrame with canonical column names (unmapped columns keep their own names)."""
    p = Path(src)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in (".xlsx", ".xls", ".csv") and not f.name.startswith(("~$", ".")))
        if not files:
            raise FileNotFoundError(f"No Excel/CSV files in {src}")
        df = pd.concat([_to_canonical(_read_raw(str(f))) for f in files], ignore_index=True)
    else:
        df = _to_canonical(_read_raw(src))
    df = df.dropna(how="all").reset_index(drop=True)
    if "docket_id" in df:  # numeric cells read as text can carry a trailing '.0'
        df["docket_id"] = df["docket_id"].map(lambda v: re.sub(r"\.0+$", "", v.strip()) if isinstance(v, str) else v)
    for c in DEFAULT_ORDER:
        if c not in df.columns:
            df[c] = pd.NA
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in DEFAULT_ORDER:
        if c not in NUMERIC:
            df[c] = df[c].astype("object").where(df[c].notna(), None)
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df
