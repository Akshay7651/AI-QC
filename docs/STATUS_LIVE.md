# LIVE STATE (auto-generated 2026-09-30 21:09:14 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **PRESENT**
- Model files (MB): {'models/digits_cnn.npz': 0.3, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['6.95', '4.23', '4.07']
- Running jobs (train / download / qc / tests):
```
(none)
```
- Last commits:
```
ddd1884 21:08 Digit reader: whole-cell CNN (cells_cnn.npz), eval + tests
c523b1f 21:04 Heartbeat: refresh live status and progress snapshot
b35f267 20:59 Heartbeat: refresh live status and progress snapshot
f5d496c 20:54 Heartbeat: refresh live status and progress snapshot
70deef2 20:49 Heartbeat: refresh live status and progress snapshot
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
