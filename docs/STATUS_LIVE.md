# LIVE STATE (auto-generated 2026-09-30 19:38:25 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **PRESENT**
- Model files (MB): {'models/digits_cnn.npz': 0.29, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['6.06', '5.02', '8.52']
- Running jobs (train / download / qc / tests):
```
12406       08:52 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
```
- Last commits:
```
ef6c7c5 19:37 WIP: agent progress snapshot
3f00d55 19:36 WIP: agent progress snapshot
29c6b54 19:34 Add one-click setup (Windows/Linux), self-check tool, training requirements
e9b8792 19:33 Heartbeat: refresh live status and progress snapshot
ae71ca3 19:33 Track the small pretrained image model; raise heartbeat size guard to 90 MB (GitHub limit is 100 MB)
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
