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
