#!/usr/bin/env python3
"""CLAP Survey AI QC - orchestrator."""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
import data_qc
import gps_qc
import ingestion
import photo_qc
import pdf_qc
import local_media
import report
import remarks
import ckstore
import chunked
from common import CostTracker

LOCAL = {}
EST_COST = {"pdf": 0.004, "photo": 0.004}
EST_SEC_PER_CALL = 4.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLAP Survey AI QC System (offline by default: no API key needed)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", help="output Excel (default <input>_QC.xlsx; the input file is never overwritten unless --inplace)")
    p.add_argument("--inplace", action="store_true", help="write the QC columns back into the input file itself")
    p.add_argument("--mode", choices=["gps", "data", "pdf", "photo", "full"], default="full")
    p.add_argument("--agents", "--workers", dest="workers", type=int, default=None,
                   help="parallel agents (worker processes). default: min(CPU cores, 12) for --engine local")
    p.add_argument("--downloaders", type=int, default=None, help="parallel download slots (default: auto)")
    p.add_argument("--rate", type=float, default=5.0, help="max new download requests per second (polite limit)")
    p.add_argument("--cost-cap", type=float)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, help="only the first N rows (quick test); the output then has N rows. With --offset: N rows from the offset")
    p.add_argument("--offset", type=int, default=None,
                   help="start at row M (0-based) of the Excel and process --limit N rows: a batch for one PC/day. GPS/same-location checks "
                        "still look at the WHOLE file. Use the same --offset/--limit/--output with --resume to continue the batch")
    p.add_argument("--chunk-rows", type=int, default=0, metavar="N",
                   help="write the output in parts <output>_part001.xlsx ... of N rows (recommended 20000 for big runs) + one merged <output>.csv; 0 = one Excel file")
    p.add_argument("--discard-media", action="store_true",
                   help="delete each row's downloaded forms/photos (cache/pdfs, cache/photos) as soon as the row is processed; never touches --local-media or the input")
    p.add_argument("--filter-state")
    p.add_argument("--filter-dist")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-media", help="ZIP or folder of downloaded forms/photos, matched to rows by docket id or mediaID")
    p.add_argument("--api-key")
    p.add_argument("--engine", choices=["auto", "claude", "local"], default="auto",
                   help="local = free offline readers (default when there is no API key); claude = Claude Vision (needs API key)")
    p.add_argument("--risk", action="store_true", help="add Risk_Score/Risk_Reasons (on by default for --engine local, full mode)")
    p.add_argument("--no-risk", action="store_true", help="skip the risk score (saves time on very large files)")
    p.add_argument("--ml-model", help="joblib model from `learn_qc.py train`; adds ML_<label> columns")
    p.add_argument("--checkpoint", default=None,
                   help="resume store (default output/checkpoint.json -> a compact output/checkpoint.sqlite is used; an old .json checkpoint is imported by --resume)")
    p.add_argument("--autosave-sec", type=float, default=60.0, help="write the output Excel + checkpoint every N seconds (0 = only at the end)")
    p.add_argument("--serve-port", type=int, default=8765, help="live dashboard port (default 8765)")
    p.add_argument("--serve-host", default="127.0.0.1", help="127.0.0.1 = this PC only (default, safe); 0.0.0.0 = also reachable from a phone on the same Wi-Fi (exposes docket IDs/remarks to that network)")
    p.add_argument("--no-serve", action="store_true", help="do not start the live dashboard web server")
    p.add_argument("--progress", default="output/progress.json", help="progress file written every second")
    p.add_argument("--task-timeout", type=float, default=180.0, help="seconds before a stuck agent is killed and its row retried")
    p.add_argument("--engine-module", help=argparse.SUPPRESS)   # tests: module providing read_form/analyse
    p.add_argument("--inline", action="store_true", help=argparse.SUPPRESS)  # agents as threads (debug/tests)
    a = p.parse_args(argv)
    if a.offset is not None and a.offset < 0:
        p.error("--offset must be >= 0")
    if a.chunk_rows and a.chunk_rows < 1:
        p.error("--chunk-rows must be >= 1 (or 0 = off)")
    if a.checkpoint is None:
        a.checkpoint = "output/checkpoint.json" if a.offset is None else f"output/checkpoint_off{a.offset}.json"
    if a.workers is not None:
        a.workers = max(1, min(a.workers, 20))
    if not a.output:
        a.output = "output/qc_output.xlsx" if a.input.startswith(("http://", "https://")) \
            else str(Path(a.input).with_suffix("")) + "_QC.xlsx"
    if a.inplace and a.chunk_rows:
        p.error("--chunk-rows cannot be combined with --inplace")
    if a.inplace and not a.input.startswith(("http://", "https://")):
        a.output = a.input if a.input.lower().endswith(".xlsx") else str(Path(a.input).with_suffix(".xlsx"))
    return a


def load_checkpoint(path):
    try:
        ck = json.loads(Path(path).read_text())
        if not isinstance(ck, dict) or not isinstance(ck.get("results"), dict):
            raise ValueError("bad checkpoint")
        ck["cost"] = float(ck.get("cost") or 0.0)
        return ck
    except (OSError, ValueError, TypeError):
        return {"results": {}, "cost": 0.0}


def save_checkpoint(path, ck):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    path = str(path)
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(ck, default=str))
    os.replace(tmp, path)


def _blank(v):
    return v is None or (isinstance(v, float) and v != v) or str(v).strip() == ""


def make_keys(df):
    """Stable per-row checkpoint keys that survive --filter-* changes and re-ordering.

    Keyed on docket_id (later duplicates of a docket get '#2', '#3', ...). Rows with no docket
    fall back to a hash of identifying fields. Positional indices are deliberately not used.
    """
    seen, keys = {}, []
    cols = [c for c in ("application_no", "farmer_name", "khasra_number", "division_number", "village",
                        "latitude", "longitude", "pdf_url", "media_urls") if c in df]
    for pos, rec in enumerate(df[["docket_id"] + cols].to_dict("records")):
        d = rec["docket_id"]
        if _blank(d):
            base = "nodocket-" + hashlib.sha1("|".join(str(rec[c]) for c in cols).encode()).hexdigest()[:16]
        else:
            base = str(d).strip()
        seen[base] = seen.get(base, 0) + 1
        keys.append(base if seen[base] == 1 else f"{base}#{seen[base]}")
    return keys


def _lookup(ck, key, pos, docket):
    """Result dict for a row; falls back to legacy '<position>|<docket>' keys from old checkpoints."""
    res = ck["results"]
    if key in res:
        return res[key]
    return res.get(f"{pos}|{docket}", {})


def _truthy_done(series):
    return series.astype(str).str.strip().str.upper().isin(["TRUE", "YES", "Y", "1"])


async def run_ai(kind, fn, df, todo, ck, tracker, args, api_key, keys=None):
    import httpx
    keys = keys or make_keys(df)
    records = df.to_dict("records")
    if args.engine == "local":
        client = None
    else:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(args.workers)
    bar = tqdm(total=len(todo), desc=f"{kind.upper():5s}", unit="row")
    done_since = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.workers * 2)) as http:
        async def one(i):
            nonlocal done_since
            async with sem:
                if tracker.exceeded:
                    return
                row = dict(records[i])
                lm = LOCAL.get(str(row.get("docket_id")).strip(), {})
                row["_local_pdfs"], row["_local_photos"] = lm.get("pdfs", []), lm.get("images", [])
                try:
                    res = await fn(row, client, http, tracker)
                except Exception as e:  # one bad row must never lose the whole run
                    res = {f"{kind}_status": "Error", f"{kind}_error": f"{type(e).__name__}: {e}"[:200]}
                ck["results"].setdefault(keys[i], {}).update(res)
                bar.update(1)
                bar.set_postfix(cost=f"${tracker.spent:.2f}")
                done_since += 1
                if done_since >= C.CHECKPOINT_EVERY:
                    done_since = 0
                    ck["cost"] = tracker.spent
                    save_checkpoint(args.checkpoint, ck)
        try:
            await asyncio.gather(*(one(i) for i in todo))
        finally:  # also on Ctrl-C / cancellation: keep everything done so far
            bar.close()
            ck["cost"] = tracker.spent
            save_checkpoint(args.checkpoint, ck)
    if tracker.exceeded:
        print(f"!! Cost cap ${tracker.cap:.2f} reached during {kind}; re-run with --resume to continue.")


def assemble(df, ck, ran_ai, keys=None):
    """Merge checkpointed AI results into the output columns."""
    keys = keys or make_keys(df)
    dockets = df["docket_id"].tolist()
    R = [_lookup(ck, k, i, dockets[i]) for i, k in enumerate(keys)]
    yn = lambda v: None if v is None else ("Yes" if v else "No")
    g = lambda k: [r.get(k) for r in R]
    out = df.copy()

    def put(col, vals):
        """Write AI values, but keep any value already in the input where AI produced none."""
        new = pd.Series(vals, index=out.index, dtype=object)
        new = new.where(new.notna(), None)
        out[col] = new.where(new.notna(), out[col].astype(object)) if col in out else new
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attempted = [bool(r) for r in R]
    if ran_ai:
        put(C.COL_DONE_BY, ["AI-Auto" if a else None for a in attempted])
        put(C.COL_QC_TIME, [now if a else None for a in attempted])
    put(C.COL_FORM_AREA, g("form_area"))
    put(C.COL_FORM_LOSS, g("form_loss"))
    put(C.COL_MATCH, [r.get("match", "NA") if r else None for r in R])
    put(C.COL_PHOTO_DATE, g("photo_date"))
    put(C.COL_FIELD_PHOTO, g("field_photo"))
    put(C.COL_SURVEYOR_SIG, [yn(r.get("surveyor_signed")) for r in R])
    put(C.COL_FARMER_SIG, [yn(r.get("farmer_signed")) for r in R])
    put(C.COL_GOVT_SIG, [yn(r.get("govt_signed")) for r in R])
    put(C.COL_FORM_STATUS, g("form_status"))
    put(C.COL_FORM_REMARKS, g("form_remarks"))
    put(C.COL_FARMER_PHOTO, [yn(r.get("farmer_photo")) for r in R])
    put(C.COL_PHOTO_LOSS, g("photo_loss"))
    conf, flags, notes, done = [], [], [], []
    for r in R:
        f = list(r.get("photo_flags") or [])
        n = []
        pc = r.get("pdf_confidence")
        if r.get("pdf_status") in ("Unavailable", "Error"):
            n.append(f"PDF {r['pdf_status'].lower()}: {r.get('pdf_error')}")
        if r.get("photo_status") in ("Unavailable", "Error"):
            n.append(f"Photos {r['photo_status'].lower()}: {r.get('photo_error')}")
        if r.get("manual_review"):
            f.append("Low OCR confidence - manual review")
        if r.get("photo_quality") in ("blurry", "dark", "irrelevant"):
            f.append(f"Photo quality: {r['photo_quality']}")
        if r.get("form_status") in ("incomplete", "overwrite"):
            f.append(f"Form {r['form_status']}")
        if r.get("match") == "Mismatch":
            f.append("Form vs app mismatch")
        failed = any(r.get(k) in ("Unavailable", "Error") for k in ("pdf_status", "photo_status"))
        if not r:
            conf.append(None)
        elif failed or (pc is not None and pc < C.LOW_CONFIDENCE_THRESHOLD) or r.get("photo_quality") == "irrelevant":
            conf.append("Low")
        elif (pc is not None and pc < 0.8) or r.get("photo_quality") in ("blurry", "dark") or f:
            conf.append("Medium")
        else:
            conf.append("High")
        flags.append(", ".join(dict.fromkeys(f)))
        notes.append("; ".join(n))
        done.append(bool(r) and not failed)
    if ran_ai:
        put(C.COL_QC_DONE, [d or None for d in done])
        put(C.COL_OTHER_REMARKS, [n or None for n in notes])
        has = pd.Series(attempted, index=out.index)
        for col, vals in (("AI_Confidence", conf), ("AI_Flags", flags)):
            new = pd.Series(vals, index=out.index, dtype=object)
            # rows with no fresh result (skipped on --resume / input already QC'd) keep their old value
            fallback = out[col].astype(object) if col in out else pd.Series("" if col == "AI_Flags" else None, index=out.index, dtype=object)
            out[col] = new.where(has, fallback)
    return out


# ============================================================================ local (offline) engine: output assembly
QC_BLOCK = [
    C.COL_DONE_BY, C.COL_QC_DONE, C.COL_QC_TIME, "QC Verdict", "AI_Confidence",
    "Form No", "PO ID matches docket", C.COL_FORM_AREA, C.COL_FORM_LOSS, "Form Row Area %", "Form Row Loss %",
    "Form Total Row Blank", "Sowing date (Form)", "Loss date (Form)", "Intimation date (Form)", "Inspection date (Form)", C.COL_MATCH, "Form vs App (Match/Mismatch/NA)",
    C.COL_SURVEYOR_SIG, C.COL_FARMER_SIG, "Primary Worker Signature (Yes/No)", C.COL_GOVT_SIG, "Officer Stamp Only (Yes/No)",
    C.COL_FORM_STATUS, "Form Quality", "Form Confidence", C.COL_FORM_REMARKS,
    C.COL_PHOTO_DATE, C.COL_FIELD_PHOTO, "Field photo type", "Photo is form image (Yes/No)", "Form-image photos (n)",
    "Duplicate photos (n)", "Photos rotated (Yes/No)", *(["Photo GPS distance (m)"] if C.USE_PHOTO_GPS else []), "Photos analysed (n)",
    "Photo scene type", "Crop present in photo", "Crop seen in photo", "Crop matches declared", "Flooding/waterlogging seen",
    "Crop damage state",
    C.COL_FARMER_PHOTO, "Farmer/person present in photos (remark)", C.COL_PHOTO_LOSS, "AI_Flags", "Same Location Remark", C.COL_OTHER_REMARKS, "AI Engine",
]
_AI_PREFIX = ("OK:", "REJECT-EVIDENCE:", "MANUAL-CHECK:", "REVIEW:", "FORM:", "PHOTOS:", "GPS:", "DATA:", "RISK ")


def _yn(v):
    return None if v is None else ("Yes" if v else "No")


def _nn(v):
    return None if v is None or (isinstance(v, float) and v != v) else v


def _form_status(form, res):
    """correct / incomplete / overwrite (None when the form could not be read).

    incomplete = the farmer or company signature is missing, or nothing is written in the value table.
    The primary worker and the block officer almost never sign (officer 0 of 149 forms), so they do not count.
    Unreadable handwriting is NOT 'incomplete'. 'overwrite' needs a real detector (none yet): it only appears if the
    form reader ever sets overwrite_suspected. Whitener / other correction types are not detected yet.
    """
    if not form or form.get("_state") != "ok" or form.get("is_proforma3") is False:
        return None
    if form.get("overwrite_suspected"):
        return "overwrite"
    cells = form.get("_cells") or {}
    if not cells:
        return None
    written = any(not c.get("blank", True) for k, c in cells.items() if k.startswith(("area_", "loss_")))
    sig_missing = form.get("farmer_signed") is False or form.get("company_signed") is False
    return "incomplete" if (sig_missing or not written) else "correct"


def person_remark(p):
    """Remark on whether a human being (the farmer, most likely) is visible in the docket's FIELD photos.
    Counts only photos that are not pictures of the paper form. A face detector is used: it finds people facing the camera
    (about 6 of 7 in testing); distant people or people seen from behind can be missed, so 'no person detected' is not proof."""
    each = p.get("person_each")
    if not isinstance(each, (list, tuple)) or not each:
        return None
    isf = p.get("photo_is_form_each") or [False] * len(each)
    field = [i for i in range(len(each)) if not (i < len(isf) and isf[i])]
    if not field:
        return "Not checked - all photos are pictures of the paper form (no field photo)"
    seen = [i + 1 for i in field if each[i]]
    n_all = len(each)
    if seen:
        return f"Farmer available - person visible in photo {', '.join(map(str, seen))} of {n_all} (not verified who it is)"
    return f"Farmer not available - no person with at least 25% of the body visible in the {len(field)} field photo(s) of {n_all}"


def _date_txt(v):
    d = remarks._parse_date(v)
    return d.strftime("%d-%m-%Y") if d else (None if v in (None, "") else str(v))


def assemble_local(df, results, keys):
    """Merge offline-engine results into the output frame: input columns + gps/data/risk + the QC block (see docs/RUN_GUIDE.md)."""
    n = len(df)
    R = results.get_many(keys) if hasattr(results, "get_many") else [results.get(k) for k in keys]
    R = [r if r and "verdict" in r else None for r in R]
    cols = {c: [None] * n for c in QC_BLOCK}
    slr = df["Same_Location_Remark"].tolist() if "Same_Location_Remark" in df else None
    for i, r in enumerate(R):
        if r is None:
            continue
        f, p = r.get("form") or {}, r.get("photo") or {}
        fok, pok = f.get("_state") == "ok", p.get("_state") == "ok"
        put = lambda c, v: cols[c].__setitem__(i, v)
        put(C.COL_DONE_BY, "AI-Auto")
        put(C.COL_QC_DONE, bool(fok or pok))
        put(C.COL_QC_TIME, r.get("ts"))
        put("QC Verdict", r.get("verdict"))
        put("AI_Confidence", r.get("confidence"))
        put("AI_Flags", ", ".join(r.get("flags") or []))
        put(C.COL_OTHER_REMARKS, r.get("remark"))
        put("Same Location Remark", _nn(slr[i]) if slr is not None else None)
        put("AI Engine", r.get("engine"))
        if f and f.get("_state") not in ("skipped",):
            put("Form vs App (Match/Mismatch/NA)", r.get("match"))
            put(C.COL_MATCH, r.get("match"))
        if fok:
            ok3 = f.get("is_proforma3") is not False
            put("Form No", f.get("form_no") if ok3 else None)
            put("PO ID matches docket", _yn(f.get("po_id_matches")) if ok3 else None)
            put(C.COL_FORM_AREA, _nn(r.get("form_area")))
            put(C.COL_FORM_LOSS, _nn(r.get("form_loss")))
            put("Form Row Area %", _nn(f.get("row_area")))
            put("Form Row Loss %", _nn(f.get("row_loss")))
            put("Form Total Row Blank", _yn(f.get("total_row_blank")))
            for col_, key_ in (("Sowing date (Form)", "sow_date"), ("Loss date (Form)", "loss_date"),
                               ("Intimation date (Form)", "intimation_date"), ("Inspection date (Form)", "inspection_date")):
                v_ = f.get(key_)
                put(col_, f"{str(v_)[:2]}-{str(v_)[2:4]}-{str(v_)[4:8]}" if v_ and len(str(v_)) == 8 and str(v_).isdigit() else None)
            put(C.COL_SURVEYOR_SIG, _yn(f.get("company_signed")))
            put(C.COL_FARMER_SIG, _yn(f.get("farmer_signed")))
            put("Primary Worker Signature (Yes/No)", _yn(f.get("worker_signed")))
            put(C.COL_GOVT_SIG, _yn(f.get("officer_signed")))
            put("Officer Stamp Only (Yes/No)", _yn(f.get("officer_stamp_only")))
            put(C.COL_FORM_STATUS, _form_status(f, r))
            put("Form Quality", f.get("quality"))
            put("Form Confidence", _nn(f.get("confidence")))
            put(C.COL_FORM_REMARKS, "; ".join(str(x) for x in (f.get("notes") or [])) or None)
        if pok:
            fp = "form image" if p.get("photo_is_form") else p.get("field_photo")
            put(C.COL_FIELD_PHOTO, fp)
            put("Field photo type", fp)
            put(C.COL_PHOTO_DATE, _date_txt(p.get("stamp_date") or p.get("photo_date")))
            put("Photo is form image (Yes/No)", _yn(bool(p.get("photo_is_form"))))
            put("Form-image photos (n)", p.get("n_form_photos"))
            put("Duplicate photos (n)", p.get("n_duplicates"))
            put("Photos rotated (Yes/No)", _yn(bool(p.get("rotated"))))
            d = _nn(p.get("stamp_dist_m"))
            if C.USE_PHOTO_GPS:
                put("Photo GPS distance (m)", None if d is None else round(float(d), 1))
            put("Photos analysed (n)", r.get("n_photos"))
            put("Photo scene type", p.get("scene_type"))
            put("Crop present in photo", _yn(p.get("crop_present")))
            put("Crop seen in photo", p.get("crop_seen"))
            put("Crop matches declared", _yn(p.get("crop_matches_declared")))
            put("Flooding/waterlogging seen", _yn(p.get("flooded")))
            put("Crop damage state", p.get("damage_state"))
            put(C.COL_FARMER_PHOTO, _yn(p.get("farmer_photo")) if p.get("farmer_photo") is not None else None)
            put("Farmer/person present in photos (remark)", person_remark(p))
            put(C.COL_PHOTO_LOSS, p.get("photo_loss"))
    out = df.copy()
    has = pd.Series([r is not None for r in R], index=out.index)
    for c in QC_BLOCK:
        new = pd.Series(cols[c], index=out.index, dtype=object)
        if c in out:
            old = out[c].astype(object)
            if c == C.COL_OTHER_REMARKS:  # keep a human's own remark (but not an earlier AI remark) after ours
                keep = old.map(lambda v: isinstance(v, str) and v.strip() != "" and not v.strip().upper().startswith(_AI_PREFIX))
                new = new.where(~(keep & has), new.astype(str) + " | INPUT REMARK: " + old.astype(str))
            new = new.where(has, old)
            if c not in ("QC Verdict", "AI_Confidence", "AI_Flags", "AI Engine", C.COL_OTHER_REMARKS):
                new = new.where(new.notna() | ~has, old)
        elif c == "AI_Flags":
            new = new.where(has, "")
        out[c] = new
    first = [c for c in out.columns if c not in QC_BLOCK]
    return out[first + QC_BLOCK]


class Autosaver:
    """Every `interval` s (and on demand) write the Excel atomically + the resume checkpoint."""

    def __init__(self, target, build, interval, ck=None, ck_path=None, lock=None, prog=None):
        self.target, self.build, self.interval = target, build, interval
        self.ck, self.ck_path, self.lock, self.prog = ck, ck_path, lock or threading.RLock(), prog
        self._stop = threading.Event()
        self._thread = None
        self._save_lock = threading.Lock()
        self.saves = 0
        self.last_error = None
        self.last_df = None

    def save(self, reason="autosave"):
        with self._save_lock:
            t = time.time()
            try:
                self.target.final = reason != "autosave"
                df = self.build()
                path = self.target.write(df)
                self.last_df = df
                if self.ck is not None and not isinstance(self.ck["results"], dict):
                    self.ck["results"].meta_set("cost", self.ck.get("cost", 0.0))    # sqlite store: rows are already committed one by one
                elif self.ck is not None and self.ck_path:
                    with self.lock:
                        snap = {"results": dict(self.ck["results"]), "cost": self.ck.get("cost", 0.0)}
                    save_checkpoint(self.ck_path, snap)
                self.saves += 1
                self.last_error = None
                if self.prog:
                    self.prog.saved(path, getattr(self.target, "rows_saved", 0) or len(df), self.target.last_note)
                    if self.target.last_note:
                        self.prog.error(self.target.last_note)
                return path
            except Exception as e:  # a failed save must not stop the run; the next tick retries
                self.last_error = f"{type(e).__name__}: {e}"
                if self.prog:
                    self.prog.error(f"{reason} failed: {self.last_error}")
                print(f"!! {reason} failed: {self.last_error}")
                return None
            finally:
                self.last_seconds = time.time() - t

    def _loop(self):
        wait = self.interval
        while not self._stop.wait(wait):
            self.save("autosave")
            # a very large sheet can take a while to write: never spend more than ~1/3 of the time saving
            wait = max(self.interval, 3 * getattr(self, "last_seconds", 0))

    def start(self):
        if self.interval and self.interval > 0:
            self._thread = threading.Thread(target=self._loop, name="autosave", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


def _start_progress(args, total, engine_label):
    import progress as P
    prog = P.Progress(path=args.progress, total=total, output_path=args.output, engine=engine_label,
                      title="CLAP AI QC - live run")
    prog.start(None if args.no_serve else args.serve_port, args.serve_host)
    if not args.no_serve:
        if prog.url:
            ip = P.lan_ip()
            print(f"Live dashboard: {prog.url}" + (f"   (phone on same Wi-Fi: http://{ip}:{prog.port}/)" if ip and args.serve_host == "0.0.0.0" else ""))
        else:
            print("!! live dashboard could not start (port busy?) - progress is still written to", args.progress)
    return prog


class _ColsView:
    """Row access over column arrays without building 160,000 dicts up front (rows[i] builds one dict on demand)."""

    def __init__(self, df, cols):
        self.cols = [c for c in cols if c in df]
        self.data = {c: df[c].to_numpy(dtype=object) for c in self.cols}
        self.n = len(df)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {c: self.data[c][i] for c in self.cols}


class _RowsView(_ColsView):
    def __init__(self, df, need):
        super().__init__(df, need)
        self.need = need

    def __getitem__(self, i):
        r = {c: self.data[c][i] for c in self.cols}
        for k in self.need:
            r.setdefault(k, None)
        return r


def _main_local(args, df, modes, ck, keys, prior_done, sl=None):
    import local_engine
    workers = args.workers or min(os.cpu_count() or 4, 12)
    kinds = [k for k, m in (("form", "pdf"), ("photo", "photo")) if m in modes]
    for step, m in enumerate(modes, 1):
        if m in ("pdf", "photo"):
            continue
        print(f"Phase {step}/{len(modes)}: {m}")
        t0 = time.time()
        if m == "data":
            df["Data_QC_Flags"] = data_qc.run(df)
        elif m == "gps":
            g = gps_qc.run(df)
            for c in g.columns:
                df[c] = g[c]
        print(f"  done in {time.time() - t0:.1f}s")
    if (args.risk or (args.mode == "full" and not args.no_risk)) and not args.no_risk:
        import learn_qc
        t0 = time.time()
        print("Phase: risk score")
        df = learn_qc.add_risk(df)
        print(f"  done in {time.time() - t0:.1f}s")
    if args.ml_model:
        import learn_qc
        df = learn_qc.add_predictions(df, args.ml_model)
    if sl is not None:   # --offset/--limit: the checks above saw the whole file; now keep only this batch
        lo, hi = sl
        df = df.iloc[lo:hi].reset_index(drop=True)
        keys, prior_done = keys[lo:hi], prior_done.iloc[lo:hi].reset_index(drop=True)
        print(f"Batch: rows {lo:,}..{lo + len(df) - 1:,} of the file ({len(df):,} rows)")
    n = len(df)
    store = ck["results"]
    stored_done = store.done_keys()
    pd_flags = prior_done.to_numpy()
    done_flags = np.array([(keys[i] in stored_done) or bool(pd_flags[i]) for i in range(n)], dtype=bool)
    done_before = int(done_flags.sum())
    todo = np.flatnonzero(~done_flags).tolist() if kinds else []
    engine_label = args.engine_module or "local"
    prog = _start_progress(args, n if kinds else 0, engine_label)
    prog.preload_done(done_before if kinds else 0)
    lock = threading.RLock()
    chunk = None
    if args.chunk_rows:
        def build_part(lo, hi):
            return assemble_local(df.iloc[lo:hi], store, keys[lo:hi])
        chunk = chunked.ChunkedOutput(args.output, args.chunk_rows, n, store, build_part, done_flags if kinds else np.ones(n, bool),
                                      needs_ai=bool(kinds), prog=prog)
        target, build = chunk, (lambda: None)
        prog.output_path = chunk.stem + "_partNNN.xlsx"
        print(f"Chunked output: {chunk.nparts} part(s) of {args.chunk_rows:,} rows -> {chunk.stem}_part001.xlsx ... and {chunk.csv_path}")
    else:
        target = report.ExcelTarget(args.output)
        build = lambda: assemble_local(df, store, keys)
    saver = Autosaver(target, build, args.autosave_sec, ck, args.checkpoint, lock, prog)
    rc, runner = 0, None
    try:
        if kinds and todo:
            gcols = [c for c in dict.fromkeys(list(gps_qc.OUT) + ["Same_Location_Remark"]) if c in df]
            ctx = {"gps": _ColsView(df, gcols) if gcols else None,
                   "dflags": df["Data_QC_Flags"].fillna("").tolist() if "Data_QC_Flags" in df else None,
                   "risk": _ColsView(df, ["Risk_Score", "Risk_Reasons"]) if "Risk_Score" in df else None}
            need = list(dict.fromkeys(list(local_engine.ROW_FIELDS)))
            rows = _RowsView(df, need)
            media = _local_map(df)
            runner = local_engine.LocalRunner(
                rows, keys, todo, ck, kinds, media, ctx, agents=workers, downloaders=args.downloaders, rate=args.rate,
                engine_module=args.engine_module, inline=args.inline, progress=prog, task_timeout=args.task_timeout,
                pause_file=str(Path(args.progress).with_name("PAUSE")), discard_media=args.discard_media,
                on_row=(lambda i, res: chunk.mark_row(i)) if chunk else None)
            runner.lock = lock
            print(f"Agents: {workers} worker processes + {runner.n_dl} downloaders  |  rows to process: {len(todo):,}"
                  f"  (already done: {done_before:,})  |  autosave every {args.autosave_sec:g}s -> {args.output}"
                  + ("  |  discarding downloaded media after each row" if args.discard_media else ""))
            saver.start()
            t0 = time.time()
            runner.run()
            print(f"Processed {runner.finalized:,} rows in {time.time() - t0:.1f}s"
                  + (f" ({runner.discarded:,} downloaded files discarded)" if args.discard_media else ""))
        else:
            saver.start()
        prog.set_status("Done")
    except KeyboardInterrupt:
        rc = 130
        print("\nStopped by user - saving what is done so far ...")
        if runner:
            runner.stop()
        prog.set_status("Stopped")
    finally:
        saver.stop()
        path = saver.save("final save")
        if prog.status == "Running":
            prog.set_status("Stopped")
        lingering = bool(prog.url) and rc == 0
        prog.write_now()
        if lingering:
            time.sleep(2.5)  # let the open dashboard page see 'Done'
        prog.stop()
    if path:
        print(f"Wrote {path}" + (f"\n!! {target.last_note}" if target.last_note else ""))
    summary = str(Path(args.output).with_name("summary_report.xlsx"))
    if chunk:
        if os.path.exists(chunk.csv_path):
            print(f"Wrote merged CSV {chunk.csv_path}")
        out = chunk.read_columns(["district", "surveyor_name", "Suggested_Remark", "Data_QC_Flags", "AI_Flags", C.COL_MATCH,
                                  C.COL_QC_DONE, "QC Verdict"])
    else:
        out = saver.last_df if saver.last_df is not None else build()
    try:
        report.build(out, summary)
        print(f"Wrote {summary}")
    except Exception as e:
        print(f"!! summary report failed: {e}")
    v = out["QC Verdict"].value_counts() if "QC Verdict" in out else {}
    print("Verdicts: " + ", ".join(f"{k}={int(v.get(k, 0))}" for k in remarks.VERDICTS) + f" | rows: {n:,} | cost: $0.00")
    if rc:
        print("Resume with the same command plus --resume")
    store.close()
    return rc


# measured on the 100-row unseen sample, 4 cores, offline engine: ~0.7-0.85 s/row wall = ~3 core-seconds per row
CORE_SEC_PER_ROW = 3.0
MB_PER_ROW_MEDIA = 1.45            # form ~0.74 MB + ~3 photos ~0.24 MB each (measured)
KB_PER_ROW_STORE = 4.6             # checkpoint (sqlite) per finished row
KB_PER_ROW_XLSX, KB_PER_ROW_CSV = 0.9, 2.3


def _fmt_dur(sec):
    sec = max(0, int(sec))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    return (f"{d} d " if d else "") + f"{h} h {r // 60:02d} min" if (d or h) else f"{r // 60} min"


def _fmt_gb(mb):
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def estimate(args, df, plan, local_map=None):
    """Numbers behind --dry-run for the offline engine (also used by the tests)."""
    from photo_qc import split_urls
    rows = sorted({i for v in plan.values() for i in v})
    n = len(rows)
    cpu = os.cpu_count() or 4
    agents = args.workers or min(cpu, 12)
    eff = min(agents, cpu)
    t_cpu = n * CORE_SEC_PER_ROW / max(eff, 1)
    net_files = 0
    if not args.local_media and rows:
        sub = df.iloc[rows]
        if "pdf" in plan:
            net_files += int(sub["pdf_url"].fillna("").astype(str).str.contains("http", regex=False).sum()) if "pdf_url" in sub else 0
        if "photo" in plan and "media_urls" in sub:
            net_files += sum(min(len(split_urls(u)), C.MAX_PHOTOS_PER_ROW) for u in sub["media_urls"].tolist())
    t_net = net_files / args.rate if (args.rate and net_files) else 0.0
    t = max(t_cpu, t_net)
    chunk = args.chunk_rows or 0
    out_mb = n * (KB_PER_ROW_XLSX + (KB_PER_ROW_CSV if chunk else 0)) / 1024
    store_mb = n * KB_PER_ROW_STORE / 1024
    inflight = max(8, agents * 4)
    media_mb = inflight * MB_PER_ROW_MEDIA if args.discard_media else (0 if args.local_media else n * MB_PER_ROW_MEDIA)
    return {"rows": n, "agents": agents, "cpu": cpu, "eff": eff, "t_cpu": t_cpu, "t_net": t_net, "net_files": net_files, "t": t,
            "sec_per_row": (t / n) if n else 0.0, "out_mb": out_mb, "store_mb": store_mb, "media_mb": media_mb,
            "chunks": (-(-n // chunk) if chunk else 1)}


def print_estimate(args, df, plan, local_map=None):
    e = estimate(args, df, plan, local_map)
    if not e["rows"]:
        print("Offline engine estimate: nothing to process.")
        return e
    print(f"Offline engine estimate for {e['rows']:,} rows ({e['agents']} agents on {e['cpu']} CPU cores; only {e['eff']} can run at once):")
    print(f"  time (CPU)      : ~{_fmt_dur(e['t_cpu'])}  (~{e['t_cpu'] / e['rows']:.2f} s/row wall; measured ~0.7-0.85 s/row on 4 cores)")
    if e["t_net"]:
        print(f"  time (download) : ~{_fmt_dur(e['t_net'])}  ({e['net_files']:,} files at --rate {args.rate:g}/s)")
    print(f"  => total        : ~{_fmt_dur(e['t'])}  (~{60 * e['rows'] / max(e['t'], 1):.0f} rows/min; add ~1 min start-up per run)"
          + ("   [limited by the download rate: raise --rate only if the site allows]" if e["t_net"] > e["t_cpu"] else ""))
    print(f"  disk: output {_fmt_gb(e['out_mb'])}" + (f" in {e['chunks']} parts + merged CSV" if args.chunk_rows else "")
          + f", checkpoint {_fmt_gb(e['store_mb'])}"
          + (f", media cache at most ~{_fmt_gb(e['media_mb'])} (--discard-media)" if args.discard_media
             else ("" if args.local_media else f", downloaded media ~{_fmt_gb(e['media_mb'])} (use --discard-media to keep it near zero)")))
    return e


def _snapshot(ck, lock):
    with lock:
        return dict(ck["results"])


def _local_map(df):
    return dict(LOCAL)


def main(argv=None):
    args = parse_args(argv)
    LOCAL.clear()  # module-level state must not leak between invocations
    print("CLAP QC System v2.0")
    df = ingestion.load(args.input)
    if args.filter_state:
        df = df[df["state"].astype(str).str.lower() == args.filter_state.lower()]
    if args.filter_dist:
        df = df[df["district"].astype(str).str.lower() == args.filter_dist.lower()]
    slice_mode = args.offset is not None
    if args.limit and not slice_mode:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    print(f"Input: {args.input} ({len(df):,} rows)")
    sl = None
    if slice_mode:
        sl = (min(args.offset, len(df)), min(len(df), args.offset + args.limit) if args.limit else len(df))
        print(f"Batch: --offset {args.offset} --limit {args.limit or 'all'} -> rows {sl[0]:,}..{sl[1] - 1:,}")
    if not args.inplace and not args.input.startswith(("http://", "https://")) \
            and Path(args.output).resolve() == Path(args.input).resolve():
        sys.exit("Refusing to overwrite the input file. Choose another --output, or add --inplace.")

    modes = {"full": ["data", "gps", "pdf", "photo"]}.get(args.mode, [args.mode])
    ai_phases = [m for m in modes if m in ("pdf", "photo")]
    keys = make_keys(df)               # keys are made on the whole file so duplicate-docket suffixes do not depend on the batch
    prior_done = _truthy_done(df[C.COL_QC_DONE]) if (args.resume and C.COL_QC_DONE in df) else pd.Series(False, index=df.index)
    full = (df, keys, prior_done)
    if sl is not None:                 # plan / dry-run / the claude engine work on the batch only
        df, keys, prior_done = (df.iloc[sl[0]:sl[1]].reset_index(drop=True), keys[sl[0]:sl[1]],
                                prior_done.iloc[sl[0]:sl[1]].reset_index(drop=True))
    api_key = args.api_key or C.ANTHROPIC_API_KEY
    if args.engine == "auto":
        args.engine = "claude" if api_key else "local"
    local_ai = args.engine == "local" and bool(ai_phases)
    if local_ai:
        if args.dry_run:      # never wipe or create a store just to estimate
            sp = ckstore.store_path_for(args.checkpoint)
            store = ckstore.open_store(args.checkpoint, True) if (args.resume and os.path.exists(sp)) else {}
        else:
            store = ckstore.open_store(args.checkpoint, args.resume)
        ck = {"results": store, "cost": 0.0}
    else:
        ck = load_checkpoint(args.checkpoint) if args.resume else {"results": {}, "cost": 0.0}
        if args.chunk_rows:
            print("Note: --chunk-rows only applies to the offline engine with forms/photos; writing one Excel file.")
    if args.local_media:
        LOCAL.update(local_media.index(args.local_media, full[0] if local_ai else df))
        print(f"Local media: matched {len(LOCAL):,} dockets "
              f"({sum(bool(v['pdfs']) for v in LOCAL.values()):,} with PDF, {sum(bool(v['images']) for v in LOCAL.values()):,} with images)")
    plan = {}
    dockets = df["docket_id"].astype(str).str.strip()
    for kind in ai_phases:
        col, doneflag = ("pdf_url", "pdf_status") if kind == "pdf" else ("media_urls", "photo_status")
        lkey = "pdfs" if kind == "pdf" else "images"
        has_remote = df[col].fillna("").astype(str).str.contains("http", regex=False)
        has_local = dockets.map(lambda d: bool(LOCAL.get(d, {}).get(lkey)))
        cand = (has_remote | has_local) & ~prior_done
        docket_list = df["docket_id"].tolist()
        plan[kind] = [i for i in np.flatnonzero(cand.to_numpy()).tolist()
                      if _lookup(ck, keys[i], i, docket_list[i]).get(doneflag) in (None, "Error")]

    if args.dry_run:
        calls = sum(len(v) for v in plan.values())
        workers = args.workers or C.MAX_PARALLEL_WORKERS
        print(f"Plan: modes={modes}")
        for k, v in plan.items():
            print(f"  {k}: {len(v):,} rows to process")
        if local_ai:
            print(f"  est. cost: ${sum(len(v) * EST_COST[k] for k, v in plan.items()):.2f} (only if you used --engine claude; the offline engine is free)")
        else:
            print(f"  est. cost: ${sum(len(v) * EST_COST[k] for k, v in plan.items()):.2f}"
                  f"   est. time: {calls * EST_SEC_PER_CALL / workers / 60:.0f} min ({workers} workers)")
        if local_ai:
            print_estimate(args, df, plan, LOCAL)
            if hasattr(ck["results"], "close"):
                ck["results"].close()
        return 0

    if ai_phases:
        print(f"OCR/photo engine: {args.engine}")
    if local_ai:
        if sl is not None:
            return _main_local(args, full[0], modes, ck, full[1], full[2], sl)
        return _main_local(args, df, modes, ck, keys, prior_done)
    if ai_phases and args.engine == "claude" and not api_key:  # fail fast, before any phase runs
        sys.exit("ANTHROPIC_API_KEY not set (use --api-key or .env, or --engine local). GPS/data modes work without it.")
    args.workers = args.workers or C.MAX_PARALLEL_WORKERS
    lock = threading.RLock()
    target = report.ExcelTarget(args.output)
    saver = Autosaver(target, lambda: assemble(df, ck, bool(ai_phases), keys), args.autosave_sec, ck, args.checkpoint, lock)
    n_phase = len(modes)
    try:
        for step, m in enumerate(modes, 1):
            print(f"Phase {step}/{n_phase}: {m}")
            t0 = time.time()
            if m == "data":
                df["Data_QC_Flags"] = data_qc.run(df)
            elif m == "gps":
                g = gps_qc.run(df)
                for c in g.columns:
                    df[c] = g[c]
            else:
                saver.start() if saver._thread is None else None
                tracker = CostTracker(args.cost_cap, ck.get("cost", 0.0))
                fn = pdf_qc.process if m == "pdf" else photo_qc.process
                asyncio.run(run_ai(m, fn, df, plan[m], ck, tracker, args, api_key, keys))
            print(f"  done in {time.time() - t0:.1f}s")
        if args.risk or args.ml_model:
            import learn_qc
            if args.risk:
                df = learn_qc.add_risk(df)
            if args.ml_model:
                df = learn_qc.add_predictions(df, args.ml_model)
    finally:
        saver.stop()
    out = assemble(df, ck, bool(ai_phases), keys)
    path = target.write(out)
    summary = str(Path(args.output).with_name("summary_report.xlsx"))
    report.build(out, summary)
    flagged = int((out.get("Suggested_Remark", pd.Series("OK", index=out.index)) != "OK").sum())
    print(f"Rows: {len(out):,} | GPS-flagged: {flagged:,} | API cost: ${ck.get('cost', 0):.2f}")
    print(f"Wrote {path} and {summary}" + (f"\n!! {target.last_note}" if target.last_note else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
