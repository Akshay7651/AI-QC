# AI-QC status / how to resume (read this first in a new session)

> Live facts (auto-refreshed every 5 minutes by `tools/status_heartbeat.py`): see **docs/STATUS_LIVE.md**.
> This file = the stable story: what we are building, why, what is done, what is next.

## Why this project exists (the goal)
The user's team QCs ~41k PMFBY (crop insurance) survey records. Each record has app-entered values (affected area %, crop loss %,
GPS) plus a photographed handwritten paper form (Proforma-3) and field photos. Human reviewers compare them by hand (slow, ~10,000 INR
already invested in this project). We are building an OFFLINE AI-QC analyst (no API key) that reads the form and photos, compares with
the app data, and writes one detailed remark + verdict per row, so humans only review the flagged rows.
Target set by the user: >95% accuracy. Our rule to honour that honestly: every value the system ASSERTS must be >=95% precise;
anything less certain is written as "not readable / low confidence - manual check" (never guess). Always measure on the hand-labelled
set in `data/eval/` and report non-zero forms separately.

## Why each part matters
- Handwritten digit reader = the single most important part: the form's affected-area/loss numbers are compared with the app values
  (mismatches are the main fraud/error signal). Tesseract cannot read this handwriting, so we train our own model on the
  ~6,000 downloaded forms, using the handwritten PO-ID (equals the known docket number) as free labels.
- Photo-is-form detection: surveyors often upload a picture of the paper form instead of a field photo (user's explicit request).
- GPS/same-location remarks: the same surveyor often surveys the same spot hundreds of times.
- Live dashboard + 60 s autosave: the user wants to watch agents work and never lose progress.
- Retraining: the user will bring ~3,000 human-QC'd rows (office laptop) to make the models better.

## IF THIS SESSION STOPS NOW - do this in the next session (in order)
1. `git pull` the branch `claude/new-session-qd93xp`; read this file and `docs/STATUS_LIVE.md`.
2. If `models/digits_cnn.npz` is missing: run `python train_digits.py --quick` (see `train_digits.py --help`); needs `data/forms/` -
   re-download with `python fetch_dataset.py --input <Level_1 Excel> --n 6000 --photo-rows 1500 --out data` (resumable; needs network access to pmfby.gov.in).
3. Evaluate: `python tools/eval_form_reader.py --cache r.pkl` -> cells/PO-ID/dates accuracy (non-zero forms separately). If <95% exact, fix segmentation/augmentation and retrain; set confidence gates so asserted values are >=95% precise.
4. Run the tests: `python -m pytest -q tests`; run a 100-row real sample: `python run_qc.py --input <excel> --limit 100 --agents 4` and look at the dashboard.
5. Write `retrain.py` + `docs/TRAINING_GUIDE.md` (see "NOT finished" below), then train on the user's 3,000 human-QC'd rows.
6. Commit small files only (<10 MB); never commit images/Excel with personal data.

Repo branch: `claude/new-session-qd93xp`. Everything below is committed unless marked otherwise.

## What exists and works (measured on the 150 hand-labelled forms in `data/eval/`)
| Piece | Files | Measured |
|---|---|---|
| Data validation, GPS clustering, same-location remark, risk score | `data_qc.py`, `gps_qc.py`, `learn_qc.py` | GPS flags match the Level-1 dashboard exactly for "Same Location"; ~6 s for 41k rows |
| Orchestrator, parallel worker processes, live dashboard, 60 s autosave, remarks + verdict | `run_qc.py`, `local_engine.py`, `remarks.py`, `progress.py`, `dashboard/live.html`, `docs/RUN_GUIDE.md` | 86+ tests pass |
| Form No under barcode | `form_reader.py` (9-pass voting) | >=3 agreeing passes: 96% exact on 73% of forms; best guess 92% on 83%; below the gate the remark says "verify manually" |
| Signatures (farmer / company / worker / officer) | `form_reader.py` | farmer 93%, company 95%, worker 92% (precision 72%), officer 98.6% (almost always empty) |
| Photo: is the "field photo" actually the paper form | `photo_local.py`, `photo_models/` | 98.5% acc (held-out 330 photos) |
| Photo: GPS/date stamp read, distance to app location | `photo_stamp.py` | parse 100%, within 200 m of app 99.9% |
| Photo: person-only vs field | `photo_local.py` | 99.5% |
| Photo: crop type (weak labels), flooded, damage state, crop present, weeds | `photo_models/heads.pkl` | crop type 92% (cotton 95%, paddy untested); flooded 97.8% but finds only 45% of floods; damage state 66%; crop present 78% (worse than always "yes"). These are only stated when both models agree with high confidence, otherwise "low confidence - manual check" (`remarks.py`) |
| Loss % from a photo | - | no signal; removed from verdicts |

## NOT finished
1. **Handwritten digit reader** (affected area %, loss %, PO ID, dates): `digits.py` + `train_digits.py` exist; `models/digits_cnn.npz` did not exist at last check (training job was still running). Until it exists `form_reader` returns None for those fields and remarks say "not readable". Train: `python train_digits.py` (labels: handwritten PO-ID = docket, plus `data/eval/cells.csv`). Then run `python tools/eval_form_reader.py --cache r.pkl` and report non-zero forms separately (109/143 labelled forms are all zero).
2. `retrain.py` (one command to retrain from a human-QC'd Excel) and `docs/TRAINING_GUIDE.md` - not written yet. Plan: human columns (Affected area% (Form), Crop Loss% (Form), signatures, Form Status, Field photo, Farmer Photo) are the labels; download evidence with `fetch_dataset.py`; hold out 15-20%; retrain digits (`train_digits.py`), photo heads (`tools/photo_train2.py`), reviewer model (`learn_qc.py train`); adopt only if better on the held-out set.
3. Final integration + timing on a quiet machine (budget <= 1.5 s/row/core), full test-suite run, and a run on a real 100-row sample.
4. Crop / flood / damage / weeds photo classes are weak (few labelled examples of cut & spread, harvested, bare soil, paddy). More hand labels help (`data/eval/photos_labels*.csv`).
5. The user will provide ~3,000 human-QC'd rows (office laptop) - use them for step 2.

## Data (not in git; re-create)
`python fetch_dataset.py --input <Level_1 Excel> --n 6000 --photo-rows 1500 --out data` then `--photos-target 8200` (forms -> `data/forms/`, photos -> `data/photos/`). Hand labels are in git under `data/eval/*.csv`. Big third-party ONNX weights are git-ignored: `python tools/fetch_models.py`.

## Gotchas learned
- pmfby.gov.in media links are public; the cloud environment needed network access set to allow `pmfby.gov.in`.
- Set `OMP_THREAD_LIMIT=1` for every Tesseract process; never use `pkill -f` (kills your own shell); kill stray tesseract processes (they hang and eat CPU).
- `survey_start_date`/`survey_end_date` in the export are the LOSS date and the farmer INTIMATION date, not the survey date (the real survey date is the committee-inspection date on the form).
- 76% of labelled forms are all-zero: always report accuracy on non-zero forms separately.
- Keep model files in git small (<10 MB each).


## FINAL ACCEPTANCE TEST (requested by the user; do this once everything is built)
1. Draw 100 random rows from the 41,377 that are NOT in `data/forms/` (training) and NOT in `data/eval/` (hand labels) - truly unseen.
2. Download their forms/photos (`fetch_dataset.py`-style, ~2 min), then establish ground truth by eye (same method as the 10-row sample in
   `output/sample10_OUTPUT.xlsx`; use `tools/make_label_sheets.py` contact sheets; spot-check with a second reader).
3. Run the normal command: `python run_qc.py --input sample100.xlsx --local-media <dir> --agents 4` (offline engine).
4. Compare field by field: Form No, PO-ID match, form area/loss and Match/Mismatch vs app, 4 signatures, photo-is-form, photo GPS/date,
   crop/flood/weeds, final verdict and remark correctness. Report accuracy AND the share of rows sent to manual check, per field.
5. Report honestly every field below 95% and why; fix and re-test on a fresh 100 if needed.
