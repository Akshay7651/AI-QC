# AI-QC Run Guide (simple, step by step)

This program checks every crop-loss survey row **on your own computer, with no internet AI and no API key**:
it reads the signed form (Proforma-3), analyses the field photos, checks GPS/data rules, scores risk, and writes
**one Excel file** with all your original columns plus the QC columns and a detailed remark for every row.
While it runs you can watch a **live dashboard** in the browser (also on your phone).

---------------------------------------------------------------------------------------------------

## 1. One-time setup

### Windows
1. **Python 3.11 or newer** - download from python.org. On the first screen tick **"Add python.exe to PATH"**, then Install.
2. **Tesseract OCR with Hindi** - download the Windows installer from the "UB Mannheim" Tesseract page. In the installer open
   *Additional language data* and tick **Hindi**. Keep the default folder (`C:\Program Files\Tesseract-OCR`).
   Then add that folder to PATH (Start menu > "Edit the system environment variables" > Environment Variables > Path > New).
3. **Get the code**: download the project ZIP from GitHub and unzip it (for example to `C:\AI-QC`), or `git clone` it.
4. Open **Command Prompt** in that folder (type `cmd` in the folder's address bar) and run:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Linux (Ubuntu / Debian)
```
sudo apt install python3 python3-venv tesseract-ocr tesseract-ocr-hin tesseract-ocr-eng
git clone <repo-url> AI-QC && cd AI-QC
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Check it works: `tesseract --list-langs` must show `eng` and `hin`.

---------------------------------------------------------------------------------------------------

## 2. Put your files in place

* The **input Excel** (for example `Level_1_GEO_Tagged_QC_Done.xlsx`): anywhere, e.g. in the project folder.
  It is **never modified** (unless you add `--inplace`).
* **Forms and photos** - two options:
  * **Nothing to do** if your PC can open the `pmfby.gov.in` links: the program downloads them itself (politely, about
    5 requests per second, files are cached in `cache/` so they are never downloaded twice).
  * If the site is blocked or slow, download them once (`python download_media.py --input X.xlsx --out downloads`, run on a
    PC that can reach the site) and pass the ZIP or folder with `--local-media downloads.zip`.
    Files are matched to rows by **docket id** (`<docket>.pdf` / `<docket>.jpg` = the form, `<docket>_1.jpg`, `<docket>_2.jpg`... = photos)
    or by the **mediaID** inside the link.

---------------------------------------------------------------------------------------------------

## 3. Run it

Always start with a **10-row test** (takes a few minutes the first time because the models load):
```
python run_qc.py --input Level_1_GEO_Tagged_QC_Done.xlsx --limit 10
```
A line like `Live dashboard: http://localhost:8765/` is printed - open it in your browser.

**Full run** (everything, all rows):
```
python run_qc.py --input Level_1_GEO_Tagged_QC_Done.xlsx
```
With your own media ZIP and a chosen number of agents:
```
python run_qc.py --input Level_1_GEO_Tagged_QC_Done.xlsx --local-media downloads.zip --agents 12
```
Output goes to `Level_1_GEO_Tagged_QC_Done_QC.xlsx` next to the input (change with `--output name.xlsx`).

**Stop and resume**: press **Ctrl-C** at any time. The program saves the Excel and a checkpoint, then stops.
Start it again with the same command plus `--resume` and it continues where it stopped (finished rows are not redone):
```
python run_qc.py --input Level_1_GEO_Tagged_QC_Done.xlsx --resume
```
(If the PC restarts or crashes, `--resume` works too - the checkpoint is saved every minute.)

**Pause without stopping**: create an empty file named `PAUSE` in the `output` folder; delete it to continue.

### Useful options
| Option | Meaning |
|---|---|
| `--limit N` | only the first N rows (test); output has N rows |
| `--agents N` (alias `--workers`) | parallel agents (default = number of CPU cores, at most 12) |
| `--local-media zip_or_folder` | use downloaded forms/photos instead of the links |
| `--output file.xlsx` | where to write (default `<input>_QC.xlsx`); `--inplace` writes into the input file |
| `--autosave-sec 60` | how often the Excel + checkpoint are saved (default every 60 s) |
| `--serve-port 8765` / `--no-serve` | live dashboard port / switch it off |
| `--serve-host 127.0.0.1` | dashboard only on this PC (default `0.0.0.0` so your phone on the same Wi-Fi can open it) |
| `--no-risk` | skip the risk score (saves about 10 s per 40,000 rows) |
| `--rate 5` | max new download requests per second (be polite to the site) |
| `--engine claude` | optional: use Claude vision instead of the offline readers (needs `ANTHROPIC_API_KEY`) |
| `--mode gps` / `data` | only the fast rule checks (no forms/photos) |

---------------------------------------------------------------------------------------------------

## 4. The live dashboard

Open **http://localhost:8765/** (on the same PC). On your **phone**: connect to the same Wi-Fi and open the second address the
program prints, like `http://192.168.1.25:8765/`. (If Windows asks about the firewall, allow Python on *private* networks.)

It shows: status (Running / Paused / Done / Stopped), progress bar, rows done, **rows per minute**, **time left**,
verdict counts, counters for each problem type (form not found, missing form link, photo is the form, GPS mismatch,
signature missing, form vs app mismatch, overwriting, crop mismatch, flooding...), one card per **agent**
(role, the docket it is working on, state, speed in seconds/row), a scrolling feed of the latest row remarks
(tap one to expand), the output file name, and the time of the last autosave. The raw data is in `output/progress.json`.

---------------------------------------------------------------------------------------------------

## 5. What is in the output Excel

All **input columns come first, unchanged** (farmer/surveyor names etc. stay as they were). Then the GPS/data/risk columns,
then the QC columns below. The sheet has a frozen header row and first column, filters, colours
(red = problem, yellow = review, green = fine) and a wide wrapped remarks column.
Rows that have not been processed yet (if you look at a partial autosave) simply have empty QC columns.

| Column | Meaning |
|---|---|
| `Data_QC_Flags` | rule-based data problems (missing values, duplicates, out-of-range, dates, bad mobile...) |
| `Nearby_Same_Surveyor_25m`, `Records_On_Same_Field`, `Group_ID`, `Cluster_Size`, `Suggested_Remark`, `Suggest_%`, `Same_Location_Remark` | GPS cluster check: how many records are within 25 m, same-field counts, suggested action |
| `Risk_Score`, `Risk_Reasons` | 0-100 relative risk priority and the top reasons |
| `Done By`, `QC Done`, `Date & Time of QC` | `AI-Auto`, whether the form or photos could be analysed, when |
| **`QC Verdict`** | **OK / Review / Reject-evidence / Manual-check** (rules in section 6) |
| `AI_Confidence` | High / Medium / Low - how much to trust the machine reading |
| **`Form No`** | the `HR0126xxxxxx` number printed under the barcode |
| **`PO ID matches docket`** | Yes/No: the handwritten PO ID equals the docket id |
| `Affected area% (Form)`, `Crop Loss% (Form)` | values from the form's **total row** (table row if the total is blank) |
| `Form Row Area %`, `Form Row Loss %`, `Form Total Row Blank` | first table row values, and whether the total row is empty |
| `Match/Mismatch (Form&app)` and `Form vs App (Match/Mismatch/NA)` | form values vs the app values (tolerance 5 points); NA = cannot compare |
| `Surveyor Signature (Yes/No)` | the **loss-assessor company** signature |
| `Farmer Signature (Yes/No)` | farmer |
| `Primary Worker Signature (Yes/No)` | primary worker (new) |
| `Government Signature (Yes/No)` | **block agriculture officer** (a rubber stamp alone is *not* a signature) |
| `Officer Stamp Only (Yes/No)` | only a stamp, no signature |
| `Form Status (correct / incomplete/ overwrite)` | overall state of the form |
| `Form Quality`, `Form Confidence`, `Survey remarks on form` | image quality, reader confidence (0-1), notes |
| `Date of survey (as per Geo Tagged Image)` | date burned into the photo |
| `Field photo (...)`, `Field photo type` | what the photos show (standing crop, no crop, cut & spread, crop mismatch, form image) |
| **`Photo is form image (Yes/No)`**, `Form-image photos (n)` | the "field photo" is actually a picture of the paper form |
| `Duplicate photos (n)`, `Photos rotated (Yes/No)` | repeated or sideways photos |
| **`Photo GPS distance (m)`** | distance between the GPS stamp on the photo and the app location |
| `Photos analysed (n)`, `Photo scene type` | how many photos, scene (field / paper form / person / house-road-sky / blurry) |
| `Crop present in photo`, `Crop seen in photo`, `Crop matches declared`, `Flooding/waterlogging seen`, `Crop damage state` | what the photo analyst saw |
| `Farmer Photo (Yes/No)`, `Loss as per Photo (Yes/No)` | person visible / estimated loss from the photo (hint only) |
| `AI_Flags` | short tags, comma separated (used by the dashboards) |
| `Same Location Remark` | text of the same-location check (several surveys at the same spot) |
| **`Any Other Remarks`** | the detailed, prioritised remark (next section) |
| `AI Engine` | which engine produced the row |

If your input already had some of these columns (for example `Done By` or `Any Other Remarks`), the AI values replace them for the
processed rows; a human's own remark is kept at the end of the new remark as `| INPUT REMARK: ...`.

---------------------------------------------------------------------------------------------------

## 6. How to read the remarks and the verdict

A remark starts with the verdict and the main reasons, then sections separated by ` | `:
```
REJECT-EVIDENCE: photographs are pictures of the form, not the field; signature missing: primary worker, block officer.
FORM: Form No HR0126175631; PO ID matches docket. Form total affected area 0% / loss 0% = app (Match).
      Signatures: farmer Yes, company Yes, primary worker NO, block officer NO. Total row blank on form.
| PHOTOS: all 5 uploaded photos are pictures of the paper form - no field photograph uploaded (photo date 27-09-2026, 12 days after inspection). GPS stamp 6 m from app.
| SAME LOCATION: ... | GPS: ... | DATA: ... | RISK 72: surveyor averages 185 records/day.
```
The text only states what was read; a value that could not be read reliably is written as **"not readable"** - never guessed.

**QC Verdict rules** (the first matching rule wins):
| Verdict | When | What to do |
|---|---|---|
| **Reject-evidence** | no signed-form link or no photo link in the record; or *all* photos are pictures of the paper form (no field photograph); or the uploaded form is not a Proforma-3 | the evidence cannot support the claim - ask the surveyor to re-upload |
| **Manual-check** | form/photos could not be downloaded or opened; form unreadable or low confidence; photos too blurry/dark | a person must look at the form/photo |
| **Review** | readable evidence but something to confirm: form differs from app, row vs total inconsistent, PO ID mismatch, overwriting, missing signature, officer stamp only, duplicate/rotated photos, photo GPS more than 200 m away, photo date outside the survey period, crop differs from declared, field looks healthy/unaffected vs reported loss, same-location surveys, data-QC flags, risk score 60 or more | quick human review |
| **OK** | none of the above | no action |

`AI_Confidence`: **Low** = evidence missing/failed or reading confidence under 0.6; **Medium** = some review flag or one of form/photos not checked;
**High** = readable form and photos and no flags.

---------------------------------------------------------------------------------------------------

## 7. How the agents (parallel workers) work

* The program starts a pool of **worker processes** ("agents"): about 60% are **Form readers**, the rest **Photo analysts**
  (an idle agent helps with the other job). Each one loads its models once, and works on one docket at a time.
* Separate **Downloader** slots fetch the forms and photos from the links (or read them from `--local-media`) **while the agents compute**, so waiting
  for the network does not slow the readers. Downloads are rate-limited politely and cached.
* Default number of agents = your CPU cores (max 12). You may ask for 10-15 with `--agents 12`; **more agents than CPU cores do not make it faster** -
  the extra ones just wait for a free core (the speed limit is the CPU, not the number of agents). Each agent uses one core on purpose
  (Tesseract would otherwise fight itself). A 4-core PC works best with `--agents 4` to `6`.
* **Crash-proof**: one bad row never stops the run. If an agent crashes or hangs (over 3 minutes on one row), it is replaced and that
  row is tried once more; if it fails again the row is marked *Manual-check* and the run continues. Such events appear under
  "Recent agent errors" on the dashboard.
* **Autosave**: every 60 seconds (and on Ctrl-C / normal end) the Excel and the checkpoint are written. The Excel is written to a temporary
  file and then swapped in, so a crash never leaves a half-written file. If you have the output Excel **open** (Windows locks it), the
  program saves to a sibling file named like `..._QC_20260930_171500.xlsx` and says so - close Excel and the next autosave goes back to the normal name.

---------------------------------------------------------------------------------------------------

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `TesseractNotFoundError` / "tesseract is not installed" | Install Tesseract (section 1) and add its folder to PATH; reopen the command window; `tesseract --version` must work. |
| Hindi text is not read | `tesseract --list-langs` must list `hin`; re-run the installer and tick Hindi. |
| "Saved to ..._QC_<time>.xlsx instead" / PermissionError | The output Excel is open. Close it (or read the timestamped copy). |
| Everything says "form could not be retrieved (HTTP 403/404)" or "connection" | The site is blocked from this PC or links expired. Download the media on a PC that can reach it (`download_media.py`) and use `--local-media`. Nothing is wrong with the program. |
| Very slow | Check the dashboard "rows / min". Close other heavy programs; use `--agents` = number of CPU cores; use `--no-risk` for huge files; the first minute is slow while the models load. A laptop usually does a few rows per second per core at best - 41,000 rows takes hours, which is why autosave/resume exist. |
| Phone cannot open the dashboard | Same Wi-Fi? Use the `http://192.168...` address printed at start; allow Python in the firewall (private network). |
| Port 8765 busy | `--serve-port 8800` (the program also tries the next ports automatically). |
| Run stopped / PC restarted | Run the same command with `--resume`. |
| Want to redo everything | Delete `output/checkpoint.json` (or run without `--resume`). |
| Excel looks empty in QC columns | Those rows are not processed yet (partial autosave) - wait for the next autosave. |

---------------------------------------------------------------------------------------------------

## 9. Using the map dashboard with the result

`dashboard/build.py` turns the output Excel into the interactive map/table dashboard: `python dashboard/build.py <output>.xlsx`
(writes `dashboard/index.html`; for big files choose the Excel with the "Choose File" button). The new columns
(`QC Verdict`, `Form No`, signatures, photo columns, ...) are recognised as QC columns automatically.
