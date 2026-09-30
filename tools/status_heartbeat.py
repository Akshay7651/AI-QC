#!/usr/bin/env python3
"""Every N seconds: refresh docs/STATUS_LIVE.md with machine-checked facts, then commit + push the branch.

  nohup python tools/status_heartbeat.py --every 300 &
Safe by design: never commits files > 90 MB (unstages them and says so), never force-pushes, skips a cycle on git errors.
"""
import argparse
import glob
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True, **kw)


def count(pattern):
    return len(glob.glob(str(ROOT / pattern)))


def facts():
    procs = sh("ps -eo pid,etime,args | grep -E 'train_digits|fetch_dataset|run_qc|photo_train|pytest' | grep -v grep | cut -c1-110").stdout.strip()
    models = {p: (round(os.path.getsize(ROOT / p) / 1e6, 2) if (ROOT / p).exists() else None)
              for p in ["models/digits_cnn.npz", "photo_models/heads.pkl", "photo_models/form.pkl", "photo_models/orient.pkl",
                        "photo_models/crop_visible.pkl", "photo_models/yunet.onnx"]}
    last = sh("git log -5 --format='%h %ad %s' --date=format:%H:%M").stdout.strip()
    load = open("/proc/loadavg").read().split()[:3] if Path("/proc/loadavg").exists() else []
    lab = {f: (sum(1 for _ in open(ROOT / f)) - 1 if (ROOT / f).exists() else None)
           for f in ["data/eval/cells.csv", "data/eval/sigs.csv", "data/eval/formno.csv", "data/eval/dates.csv", "data/eval/photos_labels.csv"]}
    return procs, models, last, load, lab


def write_live():
    procs, models, last, load, lab = facts()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    digits = "PRESENT" if models["models/digits_cnn.npz"] else "MISSING (handwritten fields unreadable until trained)"
    txt = f"""# LIVE STATE (auto-generated {now})
Read docs/STATUS.md first for the goal, the why and the resume steps. This file is only facts checked by a script.

- Handwritten digit model `models/digits_cnn.npz`: **{digits}**
- Model files (MB): {models}
- Downloaded data here (not in git): forms={count('data/forms/*.jpg')}, photos={count('data/photos/*.jpg')}
- Hand-labelled rows in git: {lab}
- Machine load (1/5/15 min): {load}
- Running jobs (train / download / qc / tests):
```
{procs or '(none)'}
```
- Last commits:
```
{last}
```
If `train_digits` is listed above it is still training: wait, do not start a second copy. If it is not listed and the model is MISSING,
the run died - restart it (STATUS.md step 2).
"""
    (ROOT / "docs" / "STATUS_LIVE.md").write_text(txt)


def commit_push():
    sh("git add -A")
    big = []
    for f in sh("git diff --cached --name-only").stdout.split():
        p = ROOT / f
        if p.exists() and p.stat().st_size > 90_000_000:
            sh(f"git reset -q HEAD -- '{f}'")
            big.append(f)
    if big:
        with open(ROOT / "docs" / "STATUS_LIVE.md", "a") as fh:
            fh.write("\nSKIPPED (too large for git): " + ", ".join(big) + "\n")
        sh("git add docs/STATUS_LIVE.md")
    if sh("git diff --cached --quiet").returncode == 0:
        return "nothing to commit"
    msg = "Heartbeat: refresh live status and progress snapshot\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01Nu3T3U4xfuBL7gnNQrdSwX"
    c = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True, text=True)
    if c.returncode:
        return "commit failed: " + c.stderr[:120]
    p = sh("git push -q origin HEAD 2>&1 | tail -1")
    return "pushed " + p.stdout.strip()[:100]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        try:
            write_live()
            print(datetime.now().strftime("%H:%M:%S"), commit_push(), flush=True)
        except Exception as e:  # never die
            print("heartbeat error:", e, flush=True)
        if a.once:
            break
        time.sleep(a.every)
