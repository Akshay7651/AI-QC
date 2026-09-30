# How to train / improve the AI-QC with your human-checked rows

You do NOT retrain for every run. Normal QC = `python run_qc.py ...` and it loads the saved models automatically.
Retrain only when you have NEW human-checked data. The more varied and the more correct, the better.

## What the models are (no .h5 files - all small, stored in git)
| File | What it does |
|---|---|
| `models/cells_cnn.npz` | reads the handwritten affected-area % and loss % in the form table (numpy file, ~1 MB) |
| `photo_models/*.pkl` | photo checks: is-it-the-form, rotation, crop / flood / damage classifiers |
| `photo_models/*.onnx` | small pretrained image model used as a feature extractor |

## Step 0 - prepare the human-QC Excel
One row per docket with at least: `Docket_ID`, `Signed_Copy_URL` (form link) and the human columns
`Affected area% (Form)` and `Crop Loss% (Form)` (plain numbers, e.g. 0, 30, 80). Keep blank if the reviewer left it blank.
Rows where the reviewer wrote something odd (text, ranges) are skipped automatically. Delete farmer names/phones if you like - not needed.
Rules to keep labels clean: every reviewer should follow the same rule for which value to copy (total row if filled, else table row 1).

## Step 1 - one command
```
python retrain.py --labels human_qc.xlsx
```
What it does: splits your dockets 85% train / 15% hold-out (the hold-out is never trained on) -> downloads the forms it does not have ->
measures the CURRENT model on the hold-out -> trains a new model (your human values override the app values) -> measures the new model
on the SAME hold-out -> **adopts it only if it is not worse**, otherwise restores the old one (a backup is kept in `models/backup/`).
Options: `--steps 6000` (more = slower, usually better), `--holdout 0.15`, `--no-download`, `--dry-run` (just validates the file and writes the split).
Time: downloading ~1 form/s; training ~30 minutes on 4 cores for 6,000 steps.

## Step 2 - check the result
`python eval_digits.py` scores the model on the 150 hand-labelled forms in `data/eval/` (kept out of all training).
Report to look at: precision when answered (target >= 95%) and coverage (share of cells it dares to answer). Uncertain cells go to manual check.

## Step 3 - use it
Nothing else to do: `python run_qc.py ...` picks up the new `models/cells_cnn.npz`. Commit and push the new model file (`git add models/cells_cnn.npz`).

## Improving the other parts
- Photo models (crop / flood / damage / weeds): add labelled photos to `data/eval/photos_labels*.csv`, then `python tools/photo_train2.py` and `python tools/photo_eval.py test`.
  Biggest gaps: paddy, cut-and-spread, harvested, bare-soil examples.
- Reviewer-pattern model (predicts how your reviewers judge a row): `python learn_qc.py train --input human_qc.xlsx --model output/model.joblib`;
  to see where the AI disagrees with your reviewers: `python learn_qc.py feedback --ai ai_output.xlsx --human human_qc.xlsx --output fb.xlsx`.
- Retraining needs PyTorch: `pip install -r requirements-train.txt`.

## How many rows are enough?
A few hundred corrected rows already help; 3,000 is a strong set. The weakest case now is NON-ZERO values (30, 60, 80 ...): make sure the file
contains plenty of them, not only all-zero forms.
