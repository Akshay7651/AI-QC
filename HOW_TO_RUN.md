# HOW TO RUN - AI-QC (step by step)

This runs fully on your own PC. No internet key, no cloud, no paid service. It uses the **CPU** (a GPU is not needed and is not used).
More CPU cores = faster. Everything below is for Windows; Linux commands are in `docs/RUN_GUIDE.md`.

---------------------------------------------------------------------------------------------------

## PART 1 - One-time setup (about 20 minutes)

**Step 1. Install Python 3.11**
Download from python.org. On the first installer screen tick **"Add Python to PATH"**.

**Step 2. Install Tesseract (reads the printed Hindi / English text)**
Download the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki and run it.
During setup, open **"Additional language data"** and tick **Hindi**. Keep the default install folder.

**Step 3. Install Git** from git-scm.com (default options).

**Step 4. Download this project**
Open the Command Prompt and run:
```
git clone https://github.com/akshay7651/ai-qc.git
cd ai-qc
git checkout claude/new-session-qd93xp
```

**Step 5. Install everything the project needs**
Double-click `setup_windows.bat`
(or run it in the Command Prompt). It creates a clean environment, installs all packages and downloads the small image model.

**Step 6. Check the setup**
```
.venv\Scripts\activate.bat
python tools\selfcheck.py
```
The last line must say **RESULT: READY**. If it says NOT READY, the lines marked FAIL tell you exactly what is missing
(for example "Hindi language data") and how to fix it.

---------------------------------------------------------------------------------------------------

## PART 2 - Put your data in place

**Step 7. The Excel file**
Your CLAP export, for example `data.xlsx` (the Level-1 file). Any column names the app uses are recognised automatically.

**Step 8. The forms and photos folder** - for example `D:\media`.
Use ONE of these layouts (the docket ID must be in the name):

1. One folder per docket (recommended for large runs):
   ```
   D:\media\040106260000636479\form\anything.jpg      <- the signed form (.jpg .png .pdf)
   D:\media\040106260000636479\media\photo1.jpg       <- field photos
   D:\media\040106260000636479\media\photo2.jpg
   ```
2. One flat folder:
   ```
   D:\media\040106260000636479.jpg                    <- the form
   D:\media\040106260000636479_1.jpg                  <- photo 1
   D:\media\040106260000636479_2.jpg                  <- photo 2
   ```
3. A folder of ZIP files, one per docket (`<docket>.zip` containing `form\` and `media\`). Works, but needs extra disk space to unpack.

Not downloaded yet? Just run without `--local-media`: the program downloads each form and photo from the link while it works (files are kept in `cache\`).
To download everything first, add `--predownload` (files go to `media\`, or `--media-dir D:\media`).

---------------------------------------------------------------------------------------------------

## PART 3 - Run it

Always start with the environment on: `.venv\Scripts\activate.bat`

**Step 9. Test on 100 rows first (about 2 minutes)**
```
python run_qc.py --input data.xlsx --local-media D:\media --limit 100 --agents 4 --output results\test100.xlsx
```
Open **http://localhost:8765** in your browser to watch the live dashboard (progress, speed, what each worker is doing).
The result is saved in `results\test100.xlsx` (it is also auto-saved every minute).

**Step 10. See how long the full run will take**
```
python run_qc.py --input data.xlsx --local-media D:\media --agents 8 --chunk-rows 20000 --dry-run
```
Set `--agents` equal to your number of CPU cores (Task Manager > Performance > CPU > "Cores"). Never more than that.
Rough speed: about 0.75 seconds per row on 4 cores -> 160,000 rows = ~33 h on 4 cores, ~17 h on 8, ~8.5 h on 16.

**Step 11. Full run (leave it overnight)**
```
python run_qc.py --input data.xlsx --local-media D:\media --agents 8 --chunk-rows 20000 --output results\all.xlsx
```
You get `results\all_part001.xlsx, all_part002.xlsx ...` (20,000 rows each) and one merged `results\all.csv` (open it in Excel).

**Step 12. If it stops (power cut, crash, you pressed Ctrl-C)**
Run the SAME command again and add `--resume`. It continues where it stopped and never redoes finished rows.

**Run on several PCs / days:** use `--offset 0 --limit 40000`, then `--offset 40000 --limit 40000` ... and merge with
`python tools\merge_qc.py results\batch*.csv --output ALL.csv`.

---------------------------------------------------------------------------------------------------

## PART 4 - Read the result

Every input row keeps all its original columns, plus the AI columns. The important ones:

| Column | Meaning |
|---|---|
| **QC Verdict** | `OK` = nothing wrong found. `Review` = a person should look. `Manual-check` = the AI could not read something reliably (blurry form, unclear handwriting). `Reject-evidence` = the evidence is not valid (no form link, or the "field photo" is really a picture of the paper form). |
| **Any Other Remarks** | One detailed sentence-by-sentence explanation: form, photos, same-location, risk. Read this first. |
| Form No | The HR0126xxxxxx number under the barcode (blank if not read reliably). |
| Affected area% (Form), Crop Loss% (Form) | Handwritten values read from the form. **Blank = "not readable, check manually"** (the AI never guesses). |
| Match/Mismatch (Form&app) | Form value compared with the app value. |
| Signature columns | Farmer / company / primary worker. The block officer is "not assessed". |
| Photo is form image (Yes/No) | Yes = all photos are a picture of the paper form (no field photo uploaded). |
| Same_Location_Remark | Multiple surveys at the same GPS spot (counts, same surveyor or not, example dockets). |
| Risk_Score / Risk_Reasons | Review priority 0-100 with reasons (a priority, not proof of anything). |
| AI_Confidence, AI_Flags | How sure the AI is and which checks fired. |

Honest accuracy (measured on 100 rows the AI had never seen): every value the AI *states* is at least 95% right
(handwritten area/loss 100%, form number 97%, signatures 94-99%, "photo is the form" 100%). The price of that safety:
it only states a handwritten value for about 57% of forms; the rest are marked **Manual-check**. More training data raises that share.

Dashboards: `dashboard\CLAP-Survey-QC.html` (manual review of AI output) and the map dashboard (`dashboard\build.py output.xlsx`).

---------------------------------------------------------------------------------------------------

## PART 5 - Make it better with your human-checked rows (retraining)

You do not retrain every run. Only when you have new human-checked data (the more varied, the better).

**Step 13.** Prepare `human_qc.xlsx` with columns `Docket_ID`, `Signed_Copy_URL`, `Affected area% (Form)`, `Crop Loss% (Form)`.
**Step 14.**
```
pip install -r requirements-train.txt
python retrain.py --labels human_qc.xlsx
```
It splits your rows 85/15 (the 15% are never trained on), trains, compares old vs new on the 15%, and **keeps the new model only if it is not worse**
(the old one is backed up). Details: `docs/TRAINING_GUIDE.md`.

---------------------------------------------------------------------------------------------------

## Problems?

| Problem | What to do |
|---|---|
| `selfcheck` says Hindi missing | Re-run the Tesseract installer, tick Hindi. |
| `tesseract is not recognized` | Add `C:\Program Files\Tesseract-OCR` to PATH, or reinstall and keep the default folder. |
| Excel says the output is locked / permission error | Close the Excel file. The program saves a timestamped copy next to it and keeps going. |
| Very slow | `--agents` too high or too low: set it to your core count. Close other heavy programs. |
| Out of disk space | Use `--discard-media` (only when it downloads by link; it never deletes your own media folder). |
| Dashboard page does not open | Another program uses port 8765: add `--serve-port 8800`, or `--no-serve`. |
| Anything else | Copy the exact error text (or a screenshot) and send it to your Claude session with the file `docs\STATUS.md`. |

More detail: `docs/RUN_GUIDE.md`, `docs/TRAINING_GUIDE.md`, `docs/STATUS.md` (current state, what is done and what is not).


---------------------------------------------------------------------------------------------------

## WORKED EXAMPLE - 10,000 rows, forms and photos already on the PC

Assumed paths (change them to yours):
- Project folder: `C:\ai-qc`            (where you ran `git clone`)
- Input Excel:    `C:\ai-qc\input\data.xlsx`
- Forms+photos:   `D:\media\<docket>\form\...jpg` and `D:\media\<docket>\media\...jpg`
- Results:        `C:\ai-qc\results\`

| Step | What it does | Command (run in `C:\ai-qc`) | Input | Output |
|---|---|---|---|---|
| 0 | Open the project, switch the environment on | `cd C:\ai-qc` then `.venv\Scripts\activate.bat` | - | - |
| 1 | Download forms/photos - **SKIP if already downloaded** | `python download_media.py --input input\data.xlsx --out D:\media` | `input\data.xlsx` (links inside) | `D:\media\<docket>.pdf` (form), `D:\media\<docket>_1.jpg` ... (photos) and also `D:\media.zip` |
| 2 | Check the PC is ready | `python tools\selfcheck.py` | - | prints `RESULT: READY` |
| 3 | Test on the first 100 rows | `python run_qc.py --input input\data.xlsx --local-media D:\media --limit 100 --agents 4 --output results\test100.xlsx` | Excel + `D:\media` | `results\test100.xlsx`, `results\summary_report.xlsx` |
| 4 | Estimate the time of the full run | `python run_qc.py --input input\data.xlsx --local-media D:\media --agents 8 --dry-run` | Excel + `D:\media` | prints estimated hours and disk use (nothing is processed) |
| 5 | **Full run, 10,000 rows** | `python run_qc.py --input input\data.xlsx --local-media D:\media --agents 8 --output results\all.xlsx` | Excel + `D:\media` | `results\all.xlsx` (all your columns + AI columns + remarks), `results\summary_report.xlsx` |
| 6 | Watch it live | open **http://localhost:8765** in the browser while step 5 runs | - | live progress page |
| 7 | If it stopped | the same command as step 5 plus `--resume` | the same | continues, never redoes finished rows |
| 8 | Map dashboard of the result | `python dashboard\build.py results\all.xlsx` then open `dashboard\index.html` | `results\all.xlsx` | `dashboard\index.html` |
| 9 | Review the flagged rows by hand | open `dashboard\CLAP-Survey-QC.html` in the browser, load `results\all.xlsx` | `results\all.xlsx` | your corrected Excel (these become your training data) |
| 10 | Improve the AI with your corrected rows | `pip install -r requirements-train.txt` then `python retrain.py --labels human_qc.xlsx` | `human_qc.xlsx` | better `models\cells_cnn.npz` (only if it is not worse) |

Notes:
- `--agents 8` means 8 parallel workers. Use your CPU core count (Task Manager > Performance > CPU > Cores), never more.
- While it runs, the file `results\all.xlsx` is saved again every minute. Close it in Excel if you open it, or it saves a timestamped copy next to it.
- The folder `output\` holds the resume state (`checkpoint...`). Do not delete it until the run is finished.
- For more than ~40,000 rows add `--chunk-rows 20000` (writes `all_part001.xlsx, all_part002.xlsx ...` and `all.csv`).
