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
