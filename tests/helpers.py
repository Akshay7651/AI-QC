import io
import zipfile

import pandas as pd

import ingestion
from column_map import DEFAULT_ORDER


def frame(rows):
    """Build a canonical DataFrame from a list of dicts (missing fields -> None)."""
    df = pd.DataFrame(rows)
    for c in DEFAULT_ORDER:
        if c not in df:
            df[c] = None
    for c in ingestion.NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.astype({c: object for c in DEFAULT_ORDER if c not in ingestion.NUMERIC})


def good(i=0, **kw):
    r = dict(docket_id=f"D{i}", farmer_name="F", crop_name="Paddy", khasra_number=str(i), division_number="0",
             village=f"V{i}", surveyor_name=f"S{i}", surveyor_mobile="9123456789", affected_area_pct=50,
             crop_loss_pct=40, total_damage_pct=20, latitude=29.0 + i * 0.01, longitude=76.0 + i * 0.01,
             survey_start_date="2026-09-01", survey_end_date="2026-09-05", district="D1", state="StateA",
             pdf_url="http://x/f.pdf", media_urls="http://x/p.jpg")
    r.update(kw)
    return r
