import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL_PDF = "claude-sonnet-5-5"
CLAUDE_MODEL_PHOTO = "claude-sonnet-5-5"

# USD per million tokens (input, output) - used only for the cost tracker / cost cap.
# Update to match your model's published pricing.
MODEL_PRICING = {"default": (3.0, 15.0)}

MAX_PARALLEL_WORKERS = 5
MAX_PHOTOS_PER_ROW = 5
PDF_RENDER_DPI = 150
PDF_MAX_PAGES = 2
PDF_TIMEOUT_SEC = 10
PHOTO_TIMEOUT_SEC = 8
MAX_IMAGE_BYTES = 5 * 1024 * 1024
CHECKPOINT_EVERY = 100
PDF_CACHE_DIR = "cache/pdfs"
PHOTO_CACHE_DIR = "cache/photos"
REFERER = "https://pmfby.gov.in"

AREA_MATCH_TOLERANCE_PCT = 5
LOSS_MATCH_TOLERANCE_PCT = 5
TOTAL_DAMAGE_TOLERANCE_PCT = 10
GPS_PROXIMITY_RADIUS_M = 25
GPS_PHOTO_MAX_DISTANCE_M = 200
SAME_FIELD_FLAG_COUNT = 3
GPS_SAME_SURVEYOR_MIN = 3   # same-surveyor neighbours within radius (>) to flag
GPS_DAMAGE_MIN = 15         # total damage % above which a hotspot is 'QC Required'
LOW_CONFIDENCE_THRESHOLD = 0.6
PHOTO_LOSS_DIFF_FLAG_PCT = 25
PHOTO_LOSS_MATCH_PCT = 20
SEASON_WINDOW_DAYS = 400

INDIA_LAT_MIN, INDIA_LAT_MAX = 8.0, 37.0
INDIA_LNG_MIN, INDIA_LNG_MAX = 68.0, 97.0

# Output column names (QC columns appended to the input schema)
COL_DONE_BY = "Done By"
COL_QC_DONE = "QC Done"
COL_QC_TIME = "Date & Time of QC"
COL_FORM_AREA = "Affected area% (Form)"
COL_FORM_LOSS = "Crop Loss% (Form)"
COL_MATCH = "Match/Mismatch (Form&app)"
COL_PHOTO_DATE = "Date of survey (as per Geo Tagged Image)"
COL_FIELD_PHOTO = "Field photo (no crop / cut & spread / crop mismatch / standing crop)"
COL_SURVEYOR_SIG = "Surveyor Signature (Yes/No)"
COL_FARMER_SIG = "Farmer Signature (Yes/No)"
COL_GOVT_SIG = "Government Signature (Yes/No)"
COL_FORM_STATUS = "Form Status (correct / incomplete/ overwrite)"
COL_FORM_REMARKS = "Survey remarks on form"
COL_FARMER_PHOTO = "Farmer Photo (Yes/No)"
COL_PHOTO_LOSS = "Loss as per Photo (Yes/No)"
COL_OTHER_REMARKS = "Any Other Remarks"

# Compare the GPS printed on each photo with the app latitude/longitude (distance column + mismatch flag). ON (user: do not switch off).
# The photo's own lat/long are NOT added as columns - the app coordinates are already in the input Excel. The photo DATE stamp is always read.
USE_PHOTO_GPS = True
