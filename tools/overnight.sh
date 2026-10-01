#!/usr/bin/env bash
# Idempotent job keeper for the long training pipeline: (re)starts every stage that is not finished and not running.
#   bash tools/overnight.sh            -> start what is missing, print a status table
# Stages (state is in files, so it is safe to call as often as you like, also after a machine restart):
#   raj_dl       download the Rajasthan QC-done forms          (done: log has DONE)
#   har_retrain  Haryana cell-model retrain on app values      (done: log has ADOPTED / NOT adopted)
#   raj_compare  blind AI run vs human values (baseline)        (done: results/compare_raj_baseline.xlsx)
cd "$(dirname "$0")/.." || exit 1
T=${T:-/tmp/t100}
running() { pgrep -f "[${1:0:1}]${1:1}" > /dev/null; }
launch() { setsid nohup "$@" > /dev/null 2>&1 < /dev/null & }

# ---- Rajasthan download
if ! grep -q DONE "$T/dl_raj2.log" 2>/dev/null && ! running "dl_raj.py"; then
  setsid nohup python "$T/dl_raj.py" >> "$T/dl_raj2.log" 2>&1 < /dev/null &
fi
# ---- Haryana retrain
if ! grep -qE "ADOPTED|NOT adopted" "$T/retrain4.log" 2>/dev/null && ! running "retrain.py --labels"; then
  rm -f data/manifest_human.csv data/holdout_human.csv
  setsid nohup python retrain.py --labels "$T/level1_minus_test.csv" --app-values --max-rows 8000 --init --steps 6000 --holdout 0.1 --procs 4 --threads 6 >> "$T/retrain4.log" 2>&1 < /dev/null &
fi
# ---- blind comparison
if [ ! -f results/compare_raj_baseline.xlsx ] && ! running "compare_human.py"; then
  mkdir -p results
  setsid nohup python tools/compare_human.py --labels "$T/raj_qc.csv" --forms data/forms_raj --out results/compare_raj_baseline.xlsx --procs 3 --cache data/compare_cache_base >> "$T/compare_base.log" 2>&1 < /dev/null &
fi
# ---- signature classifier (hand-read labels + weak labels)
if ! grep -q "^saved" "$T/train_sig_raj.log" 2>/dev/null && ! running "train_sig_raj.py"; then
  setsid nohup python tools/train_sig_raj.py --labels "$T/raj_qc.csv" --procs 2 >> "$T/train_sig_raj.log" 2>&1 < /dev/null &
fi
# ---- Rajasthan area/loss training: only after the Haryana retrain has finished (both write cell models)
if grep -qE "ADOPTED|NOT adopted" "$T/retrain4.log" 2>/dev/null && ! grep -q "^done" "$T/train_raj_final.log" 2>/dev/null && ! running "train_raj.py"; then
  setsid nohup python train_raj.py --labels "$T/raj_qc.csv" --steps 3000 --rounds 2 --procs 3 --threads 6 --init-model models/cells_cnn.npz --out models/cells_raj_new.npz >> "$T/train_raj_final.log" 2>&1 < /dev/null &
fi
sleep 1
echo "== status $(date)"
echo "raj forms on disk : $(ls data/forms_raj 2>/dev/null | wc -l)  | $(grep -c DONE $T/dl_raj2.log 2>/dev/null) done-marker"
echo "haryana retrain   : $(grep -v WARN $T/retrain4.log 2>/dev/null | tail -1)"
echo "raj compare       : $( [ -f results/compare_raj_baseline.xlsx ] && echo finished || echo 'running/not finished')"
echo "sig classifier    : $(grep -v WARN $T/train_sig_raj.log 2>/dev/null | tail -1)"
echo "raj cell training  : $(grep -v WARN $T/train_raj_final.log 2>/dev/null | tail -1)"
for p in "dl_raj.py" "retrain.py --labels" "compare_human.py" "train_sig_raj.py" "train_raj.py"; do printf "  %-22s %s\n" "$p" "$(running "$p" && echo RUNNING || echo -)"; done
