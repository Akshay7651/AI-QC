# LIVE STATE (auto-generated 2026-09-30 19:53:32 UTC)
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **PRESENT**
- Model files (MB): {'models/digits_cnn.npz': 0.29, 'photo_models/heads.pkl': 3.04, 'photo_models/form.pkl': 0.21, 'photo_models/orient.pkl': 2.42, 'photo_models/crop_visible.pkl': 0.06, 'photo_models/yunet.onnx': 0.23}
- Downloaded data here (not in git): forms=6027, photos=8204
- Hand-labelled rows in git: {'data/eval/cells.csv': 895, 'data/eval/sigs.csv': 150, 'data/eval/formno.csv': 150, 'data/eval/dates.csv': 150, 'data/eval/photos_labels.csv': 436}
- Machine load (1/5/15 min): ['8.31', '5.71', '6.13']
- Running jobs (train / download / qc / tests):
```
13861       09:48 python -u train_digits.py --rounds 4 --epochs 4 --max-forms 1500
14064       01:51 python -u -c  import train_digits as T f=T.build_cache(None, procs=3); print('cached', len(f
14085       01:50 python -u -c  import train_digits as T f=T.build_cache(None, procs=3); print('cached', len(f
14086       01:50 python -u -c  import train_digits as T f=T.build_cache(None, procs=3); print('cached', len(f
14087       01:50 python -u -c  import train_digits as T f=T.build_cache(None, procs=3); print('cached', len(f
```
- Last commits:
```
eef2ca6 19:48 Heartbeat: refresh live status and progress snapshot
06b5001 19:44 STATUS: first digit model failed; pivot to whole-cell classifier
af45d73 19:43 WIP: agent progress snapshot
fa65855 19:43 Heartbeat: refresh live status and progress snapshot
461b6b6 19:41 STATUS: train-test loop until 9/10 fresh batches >=95%
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
