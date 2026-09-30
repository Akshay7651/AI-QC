# LIVE STATE (auto-generated 2026-09-30 19:32:11 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **MISSING (handwritten fields unreadable until trained)**
- Model files (MB): {'models/digits_cnn.npz': None, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['4.01', '3.19', '10.02']
- Running jobs (train / download / qc / tests):
```
12401       02:38 /bin/bash -c source /root/.claude/shell-snapshots/snapshot-bash-1790786869885-127r4e.sh 2>/d
12406       02:38 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
12417       02:32 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
12418       02:32 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
12419       02:32 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
12420       02:32 python -u train_digits.py --rounds 3 --epochs 4 --max-forms 1500
12548       00:01 /bin/bash -c source /root/.claude/shell-snapshots/snapshot-bash-1790786869885-127r4e.sh 2>/d
12555       00:00 python3 -m pytest -q tests/test_media_and_ai.py
```
- Last commits:
```
93d6d2f 19:30 WIP: agent progress snapshot
65c9252 19:29 STATUS: scale-up requirements for 160k rows
426548f 19:27 Heartbeat: refresh live status and progress snapshot
dba6d80 19:24 STATUS: add final 100-row acceptance test plan
3a3bbc6 19:22 Heartbeat: refresh live status and progress snapshot
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
