"""Column alias registry + fuzzy matching. Nothing here knows about any state/crop."""
import re

COLUMN_ALIASES = {
    "docket_id":         ["docketID", "Docket_ID", "docket id", "docketid"],
    "application_no":    ["Application_No", "applicationNo", "app_no"],
    "farmer_name":       ["farmerName", "Farmer_Name", "farmer name"],
    "calamity_type":     ["calamityType", "Type_Of_Incidence", "typeofincidence"],
    "survey_start_date": ["surveyStartDate", "Date_Of_Incidence", "survey_date"],
    "survey_end_date":   ["surveyEndDate", "Farmer_Intimation_Date"],
    "farm_area":         ["farmArea", "Farm_Area", "farm area"],
    "crop_name":         ["cropName", "Crop_Name", "crop name"],
    "khasra_number":     ["landSurveyNumber", "Land_Survey_Number", "survey_no"],
    "division_number":   ["landDivisionNumber", "Land_Division_Number"],
    "state":             ["State", "State_Name", "state_name"],
    "district":          ["District", "District_Name", "district"],
    "tehsil":            ["Tehsil", "Level4_Name", "level4"],
    "block":             ["Patwar", "Level5_Name", "level5", "block"],
    "village":           ["Level6_Name", "level6", "village"],
    "patwar_circle":     ["Level7_Name", "level7", "patwar"],
    "surveyor_name":     ["surveyorName", "Surveyor_Name"],
    "surveyor_mobile":   ["surveyorMobile", "Surveyor_Mobile"],
    "affected_area_pct": ["Affected_Area_%", "affectedAreaPercentage", "affected area %", "Affected_area_%"],
    "crop_loss_pct":     ["Crop_Loss_%", "cropLossPercentage", "crop loss %", "Crop_loss_%"],
    "total_damage_pct":  ["Total_Damage_%", "totalDamagePercentage"],
    "surveyor_remark":   ["surveyorRemark", "Surveyor_Remark"],
    "latitude":          ["latitude", "Latitude", "lat"],
    "longitude":         ["longitude", "Longitude", "lng", "lon"],
    "pdf_url":           ["Signed_Copy_URL", "signed_copy_url", "pdf_url", "form_url"],
    "media_urls":        ["Media", "media", "media_urls", "photo_urls"],
}

# Column order of a raw, headerless CLAP export.
DEFAULT_ORDER = list(COLUMN_ALIASES.keys())

FUZZY_THRESHOLD = 92


def normalize(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("%", "pct"))


def detect_column(headers, canonical, taken=frozenset()):
    """Column index for `canonical`, or None. Exact normalized match first, then fuzzy."""
    aliases = [canonical] + COLUMN_ALIASES.get(canonical, [])
    norm_headers = [normalize(h) for h in headers]
    for alias in aliases:
        na = normalize(alias)
        for i, hn in enumerate(norm_headers):
            if i not in taken and hn == na:
                return i
    try:
        from thefuzz import fuzz
    except ImportError:
        return None
    best, best_i = 0, None
    for alias in aliases:
        na = normalize(alias)
        for i, hn in enumerate(norm_headers):
            if i in taken or not hn:
                continue
            score = fuzz.ratio(na, hn)
            if score > best:
                best, best_i = score, i
    return best_i if best >= FUZZY_THRESHOLD else None


def map_columns(headers) -> dict:
    """canonical -> column index; each source column is claimed at most once."""
    mapping, taken = {}, set()
    for canon in COLUMN_ALIASES:  # exact pass for all first, so fuzzy never steals an exact match
        idx = detect_column(headers, canon, taken) if _has_exact(headers, canon, taken) else None
        if idx is not None:
            mapping[canon] = idx
            taken.add(idx)
    for canon in COLUMN_ALIASES:
        if canon not in mapping:
            idx = detect_column(headers, canon, taken)
            if idx is not None:
                mapping[canon] = idx
                taken.add(idx)
    return mapping


def _has_exact(headers, canonical, taken):
    aliases = {normalize(a) for a in [canonical] + COLUMN_ALIASES.get(canonical, [])}
    return any(i not in taken and normalize(h) in aliases for i, h in enumerate(headers))
