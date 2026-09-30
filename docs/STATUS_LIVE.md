# LIVE STATE (auto-generated 2026-09-30 19:21:59 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **MISSING (handwritten fields unreadable until trained)**
- Model files (MB): {'models/digits_cnn.npz': None, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['1.10', '7.86', '17.56']
- Running jobs (train / download / qc / tests):
```
31538       56:06 python -u train_digits.py --quick --rounds 2 --max-forms 900
```
- Last commits:
```
9a72878 19:19 Add docs/STATUS.md: current state, measured numbers, what is left, how to resume
e8afa07 19:15 Form No: 9-pass voting with agreement-based confidence (>=3 agreeing passes -> 96% exact)
ce4f3ca 19:12 Remarks: assert photo crop/flood/damage claims only when both models agree with high confidence
37368ce 19:10 Photo analyst: crop/flood/damage/scene heads (small), dual-model check, fetch_models tool
444357b 19:08 WIP: agent progress snapshot
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
