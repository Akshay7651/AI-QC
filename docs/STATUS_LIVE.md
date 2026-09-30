# LIVE STATE (auto-generated 2026-09-30 19:33:22 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **MISSING (handwritten fields unreadable until trained)**
- Model files (MB): {'models/digits_cnn.npz': None, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['3.73', '3.28', '9.55']
- Running jobs (train / download / qc / tests):
```
12401       03:49 /bin/bash -c source /root/.claude/shell-snapshots/snapshot-bash-1790786869885-127r4e.sh 2>/d
12406       03:49 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
```
- Last commits:
```
ae71ca3 19:33 Track the small pretrained image model; raise heartbeat size guard to 90 MB (GitHub limit is 100 MB)
b339925 19:32 local_media: support per-docket ZIPs and <docket>/form + <docket>/media folder layouts; docs
188ef85 19:32 Heartbeat: refresh live status and progress snapshot
93d6d2f 19:30 WIP: agent progress snapshot
65c9252 19:29 STATUS: scale-up requirements for 160k rows
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
