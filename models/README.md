# models/ - offline handwritten-number readers for Proforma-3

| file | what | size |
|---|---|---|
| `cells_cnn.npz` | **main model**: whole-cell CNN (64x192 line-removed ink mask -> EMPTY / 0,5,...,100 / OTHER), float16, numpy inference | 0.8 MB |
| `digits_cnn.npz` | per-digit CNN (13 classes: 0-9, '.', mark, junk), used by `read_digit_string` / `read_number_segmented` (weak) | 0.3 MB |
| `metrics.json` | held-out metrics of the per-digit model (weak labels) | |
| `eval_hand_labels.json` | accuracy of `digits.read_number` on the lead's hand labels `data/eval/cells.csv` | |

Retrain: `python train_digits.py --rounds 4 --epochs 4` (per-digit model; caches crops of every `data/forms/*.jpg`
in `data/digits_cache/`), then `python train_cells.py --rebuild --steps 6000` (whole-cell model from the app values in
`data/manifest.csv`; dockets listed in `data/eval/*.csv` hand-label files are never trained on), then
`python eval_digits.py` (hand-label accuracy). Training uses MNIST only for the per-digit model.
Labels for the cell model are the APP values (weak labels; ~3% differ from the paper form) - so the model can inherit app errors.
