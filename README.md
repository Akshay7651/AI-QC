**New here? Read [HOW_TO_RUN.md](HOW_TO_RUN.md) - step-by-step guide (setup, run, read the output, retrain).**

# CLAP Survey AI QC

Data-agnostic QC for PMFBY CLAP survey exports. `pip install -r requirements.txt`, put `ANTHROPIC_API_KEY` in `.env`.

```bash
python run_qc.py --input data.xlsx --mode gps                 # free: GPS proximity (no API)
python run_qc.py --input data.xlsx --mode data                # free: validation rules
python run_qc.py --input data.xlsx --dry-run                  # plan + cost/time estimate
python run_qc.py --input data.xlsx --output out.xlsx --workers 8 --cost-cap 50 --resume
# PDFs/photos downloaded locally (ZIP or folder) - matched to rows by docket id:
python run_qc.py --input data.xlsx --local-media downloads.zip
```

Local media: any file whose path contains the docket id is used (`<docket>.pdf`, `<docket>_1.jpg`,
`<docket>/photo.jpg`). Local files are preferred; photo slots left over (max 5) are filled from URLs.
Rows with only local files (no URL in the Excel) are still QC'd.

Outputs: `<output>.xlsx` (input + QC columns) and `summary_report.xlsx`. `output/checkpoint.json` enables `--resume`.
Cost rates are in `config.MODEL_PRICING` - set them to your model's real pricing so `--cost-cap` is accurate.

## Notes from the Haryana Kharif 2026 file
- Real export headers (`Docket_ID`, `Level7_Name`, `Signed_Copy_URL`, ...) are auto-mapped; existing QC columns in the input are kept and only filled/overwritten where AI produced a value.
- GPS rules follow the Level-1 dashboard (`>3` same-surveyor points within 25 m; total damage `>15` => "QC Required", else "Low Damage"; `>3` records on a survey number => "Review - Multiple Records"). Every row the dashboard flagged as Same Location is flagged identically; we additionally flag ~4.3k rows whose cross-grid-cell neighbours the dashboard's spatial hash misses.
- `pmfby.gov.in` media URLs are only reachable from your network, so run `run_qc.py` on your PC, or download the files and pass `--local-media` (files may be named by docket id **or** by the mediaID GUID from the URL).

## No API key? Use the free local engine
`python run_qc.py --input data.xlsx --local-media downloads.zip --engine local --risk`
(`--engine auto`, the default, uses Claude only when `ANTHROPIC_API_KEY` is set.) Needs `tesseract-ocr` with the `hin` language
(`apt install tesseract-ocr tesseract-ocr-hin`, or the UB-Mannheim Windows installer). Local OCR reads area/loss/date by label and
detects signatures by ink; photo checks use EXIF/stamp GPS+date, colour and person detection. Treat crop-type/loss photo guesses as hints.
`--risk` adds `Risk_Score`/`Risk_Reasons`; `learn_qc.py train|predict|feedback` learns from human-QC'd rows once you have them.

## Quick 10-row test on your PC (no settings changes needed)
```
pip install -r requirements.txt
python download_media.py --input test10.xlsx --out downloads --limit 10
python run_qc.py --input test10.xlsx --output test10_QC.xlsx --local-media downloads.zip --engine local --risk
```
