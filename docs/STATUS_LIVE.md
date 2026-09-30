# LIVE STATE (auto-generated 2026-09-30 21:19:20 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **PRESENT**
- Model files (MB): {'models/digits_cnn.npz': 0.3, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['0.40', '2.99', '4.12']
- Running jobs (train / download / qc / tests):
```
19660       03:49 python3 fetch_dataset.py --input /tmp/claude-0/unseen100.xlsx --n 100 --photo-rows 100 --out
```
- Last commits:
```
b3f608f 21:15 STATUS: end-to-end form reader numbers after gating
33b4fed 21:14 Heartbeat: refresh live status and progress snapshot
fe08ec8 21:12 Form reader: confidence gate on handwritten area/loss (0.8), disable false overwrite rule, hide unreliable dates
266f695 21:09 Heartbeat: refresh live status and progress snapshot
ddd1884 21:08 Digit reader: whole-cell CNN (cells_cnn.npz), eval + tests
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
