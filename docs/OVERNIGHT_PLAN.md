# Overnight plan (Rajasthan + Haryana training) - for whoever resumes this session

User's request (2026-10-01, going to sleep): "do not stop until finished": run the AI on the human-QC'd (done) forms WITHOUT using their
remarks, match with the human values, analyse the differences (open the forms and see where the AI is wrong), then train properly, test and push the models.
Up to 20 agents may be launched for parallel visual work.

State is in files; `bash tools/overnight.sh` restarts whatever is unfinished and prints the status (logs in /tmp/t100/*.log).
Stages / done markers
1. `dl_raj.py` -> data/forms_raj (2,259 Rajasthan QC-done forms; log /tmp/t100/dl_raj2.log has DONE)
2. `retrain.py --app-values` Haryana model (log /tmp/t100/retrain4.log, ADOPTED / NOT adopted line). It rewrites models/cells_cnn.npz.
3. `tools/compare_human.py` baseline -> results/compare_raj_baseline.xlsx (blind AI vs human values, Rajasthan)
4. Sheets for vision labelling: `tools/make_sig_sheets_raj.py` -> data/eval_raj/sheets/sheet_NNN.jpg + index.csv; agents write data/eval_raj/sigs_<k>.csv
   (docket,farmer,company,aao with 1/0/?).  Then `tools/train_sig_raj.py --labels /tmp/t100/raj_qc.csv` (weak labels from the human status text), validated on the vision labels.
5. Mismatch review: agents open the forms listed in the compare xlsx (category WRONG / MULTI-WRONG / ABSTAIN with human value non-zero) and write
   data/eval_raj/review_<k>.csv (docket, verdict: AI_wrong / HUMAN_wrong / ROW_AMBIGUOUS / UNREADABLE, note).
6. `train_raj.py --labels /tmp/t100/raj_qc.csv --steps 3000 --rounds 2 --out models/cells_raj_new.npz --init-model models/cells_cnn.npz` (after stage 2 finished!)
   then evaluate: Haryana `OMP_THREAD_LIMIT=1 python /tmp/t100/gate_eval.py` (must keep >=95% precision at the gate) and
   `python tools/compare_human.py ...` for Rajasthan (hold-out only counts); adopt (copy to models/cells_cnn.npz), set CELL_GATE, run tests, commit + push, tell the user.
Never train two things that write models/cells_cnn.npz at the same time.

## Where things are on the forms (read by eye from the real forms)
Haryana Proforma-3 (IndusInd / Reliance GIC; barcode + printed form number HR0126xxxxxx under the barcode, top right; PO ID handwritten under it):
 header grid (rows 1-15: farmer, father, village, block, district, company, crop, ... dates in rows 11-14: sowing, loss, intimation, inspection) ->
 table (cols: no, application no, ticket, land survey / khasra, insured area, AFFECTED AREA %, LOSS %, remarks; rows 1-10 + a TOTAL row) ->
 committee description -> signatures: farmer | loss-assessor company | primary worker | khand krishi adhikari (block officer, almost never signs).
Rajasthan (National Insurance Company Ltd, Jaipur Regional Office): printed header, blank 'form number' field (not used), small grid (farmer, father, patwar halka,
 tehsil, district, farmer id, loss date, intimation date), 3 handwritten lines (harvest date, cause, committee inspection date) ->
 table of 9 columns (no, application id, DOCKET ID, crop, khasra, khata, insured area ha, AFFECTED AREA %, LOSS %), 10 rows, NO total row, several dockets may share a form
 -> cause of loss, remarks -> signatures: farmer | insurance company TC/DC | agriculture supervisor / AAO (with stamp).

## Result of Rajasthan training run (2026-10-02, new geometry)
`train_raj.py` (2 rounds x 3000 steps, 1213 train / 213 hold-out forms) -> models/cells_raj_new.npz (not committed, NOT adopted).
Blind compare on the same 213 hold-out forms: baseline Haryana model 28.2% coverage / 100% precision (8 of 84 non-zero forms stated);
new model 9.9% coverage / 100% precision (9 of 84 non-zero). Coverage is worse, so models/cells_cnn.npz is unchanged.
Note: the cloud machine only runs while a turn is active; long jobs stall between keeper ticks (stay in a turn that waits).
