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
    p.add_argument("--limit", type=int, help="only the first N rows (quick test); the output then has N rows")
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
    p.add_argument("--checkpoint", default="output/checkpoint.json")
    p.add_argument("--autosave-sec", type=float, default=60.0, help="write the output Excel + checkpoint every N seconds (0 = only at the end)")
    p.add_argument("--serve-port", type=int, default=8765, help="live dashboard port (default 8765)")
    p.add_argument("--serve-host", default="0.0.0.0", help="0.0.0.0 = reachable from your phone on the same Wi-Fi; 127.0.0.1 = this PC only")
    p.add_argument("--no-serve", action="store_true", help="do not start the live dashboard web server")
    p.add_argument("--progress", default="output/progress.json", help="progress file written every second")
    p.add_argument("--task-timeout", type=float, default=180.0, help="seconds before a stuck agent is killed and its row retried")
    p.add_argument("--engine-module", help=argparse.SUPPRESS)   # tests: module providing read_form/analyse
    p.add_argument("--inline", action="store_true", help=argparse.SUPPRESS)  # agents as threads (debug/tests)
    a = p.parse_args(argv)
    if a.workers is not None:
        a.workers = max(1, min(a.workers, 20))
    if not a.output:
        a.output = "output/qc_output.xlsx" if a.input.startswith(("http://", "https://")) \
            else str(Path(a.input).with_suffix("")) + "_QC.xlsx"
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
    "Form Total Row Blank", C.COL_MATCH, "Form vs App (Match/Mismatch/NA)",
    C.COL_SURVEYOR_SIG, C.COL_FARMER_SIG, "Primary Worker Signature (Yes/No)", C.COL_GOVT_SIG, "Officer Stamp Only (Yes/No)",
    C.COL_FORM_STATUS, "Form Quality", "Form Confidence", C.COL_FORM_REMARKS,
    C.COL_PHOTO_DATE, C.COL_FIELD_PHOTO, "Field photo type", "Photo is form image (Yes/No)", "Form-image photos (n)",
    "Duplicate photos (n)", "Photos rotated (Yes/No)", "Photo GPS distance (m)", "Photos analysed (n)",
    C.COL_FARMER_PHOTO, C.COL_PHOTO_LOSS, "AI_Flags", C.COL_OTHER_REMARKS, "AI Engine",
]
_AI_PREFIX = ("OK:", "REJECT-EVIDENCE:", "MANUAL-CHECK:", "REVIEW:", "FORM:", "PHOTOS:", "GPS:", "DATA:", "RISK ")


def _yn(v):
    return None if v is None else ("Yes" if v else "No")


def _nn(v):
    return None if v is None or (isinstance(v, float) and v != v) else v


def _form_status(form, res):
    if not form or form.get("_state") != "ok":
        return None
    if form.get("overwrite_suspected"):
        return "overwrite"
    sigs = [form.get(k) for k in ("farmer_signed", "company_signed", "worker_signed", "officer_signed")]
    if res.get("form_area") is None or any(v is False for v in sigs):
        return "incomplete"
    return "correct"


def _date_txt(v):
    d = remarks._parse_date(v)
    return d.strftime("%d-%m-%Y") if d else (None if v in (None, "") else str(v))


def assemble_local(df, results, keys):
    """Merge offline-engine results into the output frame: input columns + gps/data/risk + the QC block (see docs/RUN_GUIDE.md)."""
    n = len(df)
    R = [results.get(k) for k in keys]
    R = [r if r and "verdict" in r else None for r in R]
    cols = {c: [None] * n for c in QC_BLOCK}
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
            put("Photo GPS distance (m)", None if d is None else round(float(d), 1))
            put("Photos analysed (n)", r.get("n_photos"))
            put(C.COL_FARMER_PHOTO, _yn(p.get("farmer_photo")) if p.get("farmer_photo") is not None else None)
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

    def save(self, reason="autosave"):
        with self._save_lock:
            t = time.time()
            try:
                df = self.build()
                path = self.target.write(df)
                if self.ck is not None and self.ck_path:
                    with self.lock:
                        snap = {"results": dict(self.ck["results"]), "cost": self.ck.get("cost", 0.0)}
                    save_checkpoint(self.ck_path, snap)
                self.saves += 1
                self.last_error = None
                if self.prog:
                    self.prog.saved(path, len(df), self.target.last_note)
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
        while not self._stop.wait(self.interval):
            self.save("autosave")

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


def _main_local(args, df, modes, ck, keys, prior_done):
    import local_engine
    n = len(df)
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

    done_before = [i for i in range(n) if "verdict" in ck["results"].get(keys[i], {}) or prior_done.iloc[i]]
    done_set = set(done_before)
    todo = [i for i in range(n) if i not in done_set] if kinds else []
    engine_label = args.engine_module or "local"
    prog = _start_progress(args, n if kinds else 0, engine_label)
    prog.preload_done(len(done_before) if kinds else 0)
    lock = threading.RLock()
    target = report.ExcelTarget(args.output)
    build = lambda: assemble_local(df, _snapshot(ck, lock), keys)
    saver = Autosaver(target, build, args.autosave_sec, ck, args.checkpoint, lock, prog)
    rc, runner = 0, None
    try:
        if kinds and todo:
            gcols = [c for c in gps_qc.OUT if c in df]
            ctx = {"gps": df[gcols].to_dict("records") if gcols else None,
                   "dflags": df["Data_QC_Flags"].fillna("").tolist() if "Data_QC_Flags" in df else None,
                   "risk": df[["Risk_Score", "Risk_Reasons"]].to_dict("records") if "Risk_Score" in df else None}
            need = list(dict.fromkeys(list(local_engine.ROW_FIELDS)))
            rows = df[[c for c in need if c in df]].to_dict("records")
            for r in rows:
                for k in need:
                    r.setdefault(k, None)
            media = _local_map(df)
            runner = local_engine.LocalRunner(
                rows, keys, todo, ck, kinds, media, ctx, agents=workers, downloaders=args.downloaders, rate=args.rate,
                engine_module=args.engine_module, inline=args.inline, progress=prog, task_timeout=args.task_timeout,
                pause_file=str(Path(args.progress).with_name("PAUSE")))
            runner.lock = lock
            print(f"Agents: {workers} worker processes + {runner.n_dl} downloaders  |  rows to process: {len(todo):,}"
                  f"  (already done: {len(done_before):,})  |  autosave every {args.autosave_sec:g}s -> {args.output}")
            saver.start()
            t0 = time.time()
            runner.run()
            print(f"Processed {runner.finalized:,} rows in {time.time() - t0:.1f}s")
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
    out = build()
    try:
        summary = str(Path(args.output).with_name("summary_report.xlsx"))
        report.build(out, summary)
        print(f"Wrote {summary}")
    except Exception as e:
        print(f"!! summary report failed: {e}")
    v = out["QC Verdict"].value_counts() if "QC Verdict" in out else {}
    print("Verdicts: " + ", ".join(f"{k}={int(v.get(k, 0))}" for k in remarks.VERDICTS) + f" | rows: {len(out):,} | cost: $0.00")
    if rc:
        print("Resume with the same command plus --resume")
    return rc


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
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)
    print(f"Input: {args.input} ({len(df):,} rows)")
    if not args.inplace and not args.input.startswith(("http://", "https://")) \
            and Path(args.output).resolve() == Path(args.input).resolve():
        sys.exit("Refusing to overwrite the input file. Choose another --output, or add --inplace.")

    modes = {"full": ["data", "gps", "pdf", "photo"]}.get(args.mode, [args.mode])
    ai_phases = [m for m in modes if m in ("pdf", "photo")]
    ck = load_checkpoint(args.checkpoint) if args.resume else {"results": {}, "cost": 0.0}

    keys = make_keys(df)
    prior_done = _truthy_done(df[C.COL_QC_DONE]) if (args.resume and C.COL_QC_DONE in df) else pd.Series(False, index=df.index)
    if args.local_media:
        LOCAL.update(local_media.index(args.local_media, df))
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
        print(f"  est. cost: ${sum(len(v) * EST_COST[k] for k, v in plan.items()):.2f}"
              f"   est. time: {calls * EST_SEC_PER_CALL / workers / 60:.0f} min ({workers} workers)")
        return 0

    api_key = args.api_key or C.ANTHROPIC_API_KEY
    if args.engine == "auto":
        args.engine = "claude" if api_key else "local"
    if ai_phases:
        print(f"OCR/photo engine: {args.engine}")
    if args.engine == "local" and ai_phases:
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
