#!/usr/bin/env python3
"""CLAP Survey AI QC - orchestrator."""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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
from common import CostTracker

LOCAL = {}
EST_COST = {"pdf": 0.004, "photo": 0.004}
EST_SEC_PER_CALL = 4.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLAP Survey AI QC System")
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--mode", choices=["gps", "data", "pdf", "photo", "full"], default="full")
    p.add_argument("--workers", type=int, default=C.MAX_PARALLEL_WORKERS)
    p.add_argument("--cost-cap", type=float)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--filter-state")
    p.add_argument("--filter-dist")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--local-media", help="ZIP or folder of downloaded PDFs/photos, matched to rows by docket id")
    p.add_argument("--api-key")
    p.add_argument("--checkpoint", default="output/checkpoint.json")
    a = p.parse_args(argv)
    a.workers = max(1, min(a.workers, 20))
    if not a.output:
        a.output = str(Path(a.input).with_suffix("")) + "_QC.xlsx"
    return a


def load_checkpoint(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"results": {}, "cost": 0.0}


def save_checkpoint(path, ck):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    Path(tmp).write_text(json.dumps(ck, default=str))
    os.replace(tmp, path)


def row_key(i, row):
    return f"{i}|{row.get('docket_id')}"


async def run_ai(kind, fn, df, todo, ck, tracker, args, api_key):
    import anthropic
    import httpx
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
                row = df.loc[i].to_dict()
                lm = LOCAL.get(str(row.get("docket_id")).strip(), {})
                row["_local_pdfs"], row["_local_photos"] = lm.get("pdfs", []), lm.get("images", [])
                res = await fn(row, client, http, tracker)
                ck["results"].setdefault(row_key(i, row), {}).update(res)
                bar.update(1)
                bar.set_postfix(cost=f"${tracker.spent:.2f}")
                done_since += 1
                if done_since >= C.CHECKPOINT_EVERY:
                    done_since = 0
                    ck["cost"] = tracker.spent
                    save_checkpoint(args.checkpoint, ck)
        await asyncio.gather(*(one(i) for i in todo))
    bar.close()
    ck["cost"] = tracker.spent
    save_checkpoint(args.checkpoint, ck)
    if tracker.exceeded:
        print(f"!! Cost cap ${tracker.cap:.2f} reached during {kind}; re-run with --resume to continue.")


def assemble(df, ck, ran_ai):
    """Merge checkpointed AI results into the output columns."""
    R = [ck["results"].get(row_key(i, r), {}) for i, r in enumerate(df.to_dict("records"))]
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
        out["AI_Confidence"] = conf
        out["AI_Flags"] = flags
    return out


def main(argv=None):
    args = parse_args(argv)
    print("CLAP QC System v1.0")
    df = ingestion.load(args.input)
    if args.filter_state:
        df = df[df["state"].astype(str).str.lower() == args.filter_state.lower()]
    if args.filter_dist:
        df = df[df["district"].astype(str).str.lower() == args.filter_dist.lower()]
    df = df.reset_index(drop=True)
    print(f"Input: {args.input} ({len(df):,} rows)")

    modes = {"full": ["data", "gps", "pdf", "photo"]}.get(args.mode, [args.mode])
    ai_phases = [m for m in modes if m in ("pdf", "photo")]
    ck = load_checkpoint(args.checkpoint) if args.resume else {"results": {}, "cost": 0.0}

    prior_done = df[C.COL_QC_DONE].astype(str).str.upper().eq("TRUE") if (args.resume and C.COL_QC_DONE in df) else pd.Series(False, index=df.index)
    if args.local_media:
        LOCAL.update(local_media.index(args.local_media, df))
        print(f"Local media: matched {len(LOCAL):,} dockets "
              f"({sum(bool(v['pdfs']) for v in LOCAL.values()):,} with PDF, {sum(bool(v['images']) for v in LOCAL.values()):,} with photos)")
    plan = {}
    for kind in ai_phases:
        col, doneflag = ("pdf_url", "pdf_status") if kind == "pdf" else ("media_urls", "photo_status")
        lkey = "pdfs" if kind == "pdf" else "images"
        plan[kind] = [i for i in df.index
                      if (df.at[i, col] or LOCAL.get(str(df.at[i, "docket_id"]).strip(), {}).get(lkey)) and not prior_done[i]
                      and ck["results"].get(row_key(i, df.loc[i].to_dict()), {}).get(doneflag) in (None, "Error")]

    if args.dry_run:
        calls = sum(len(v) for v in plan.values())
        print(f"Plan: modes={modes}")
        for k, v in plan.items():
            print(f"  {k}: {len(v):,} rows to process")
        print(f"  est. cost: ${sum(len(v) * EST_COST[k] for k, v in plan.items()):.2f}"
              f"   est. time: {calls * EST_SEC_PER_CALL / args.workers / 60:.0f} min ({args.workers} workers)")
        return 0

    n_phase = len(modes)
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
            api_key = args.api_key or C.ANTHROPIC_API_KEY
            if not api_key:
                sys.exit("ANTHROPIC_API_KEY not set (use --api-key or .env). GPS/data modes work without it.")
            tracker = CostTracker(args.cost_cap, ck.get("cost", 0.0))
            fn = pdf_qc.process if m == "pdf" else photo_qc.process
            asyncio.run(run_ai(m, fn, df, plan[m], ck, tracker, args, api_key))
        print(f"  done in {time.time() - t0:.1f}s")

    out = assemble(df, ck, bool(ai_phases))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(args.output, index=False)
    summary = str(Path(args.output).with_name("summary_report.xlsx"))
    report.build(out, summary)
    flagged = int((out.get("Suggested_Remark", pd.Series("OK", index=out.index)) != "OK").sum())
    print(f"Rows: {len(out):,} | GPS-flagged: {flagged:,} | API cost: ${ck.get('cost', 0):.2f}")
    print(f"Wrote {args.output} and {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
