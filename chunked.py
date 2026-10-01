"""Chunked output for very large runs: `<output>_part001.xlsx`, `_part002.xlsx` ... plus ONE merged `<output>.csv`.

* A part is written only from the rows it covers (memory stays small), atomically, and is never rewritten once every row of
  the part is finished (a marker in the checkpoint store says so; it survives resume).
* A part that is still in progress is rewritten on every autosave only when its number of finished rows changed.
* Every written part also leaves a header-less CSV piece in `<output>_csvparts/`; the merged CSV (UTF-8 with BOM, so Excel shows
  Hindi text correctly) is a plain byte-concatenation of header + pieces, rebuilt when a part completes and at the end.
"""
from __future__ import annotations

import csv
import io
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import report


def part_count(n, chunk):
    return max(1, -(-int(n) // int(chunk)))


class ChunkedOutput:
    def __init__(self, output, chunk_rows, n_rows, store, build_part, done_flags, needs_ai=True, prog=None):
        """build_part(lo, hi) -> DataFrame for rows lo..hi-1.  done_flags: bool array (finished rows, will be updated via mark_row)."""
        out = str(output)
        self.stem = out[:-5] if out.lower().endswith(".xlsx") else out
        self.chunk, self.n = int(chunk_rows), int(n_rows)
        self.nparts = part_count(self.n, self.chunk)
        self.store, self.build_part, self.needs_ai, self.prog = store, build_part, needs_ai, prog
        self.done = np.asarray(done_flags, dtype=bool).copy()
        self.csv_dir = Path(self.stem + "_csvparts")
        self.csv_path = self.stem + ".csv"
        self.last_note = ""
        self.final = False
        self.rows_saved = 0
        self.header = None
        self._written = {}          # part -> finished-row count at the last write
        self._targets = {}
        sig = f"{self.chunk}:{self.n}"
        if store.meta_get(f"chunks:{self.stem}") != sig:      # different chunk size / row count: old parts do not apply
            store.meta_delete_prefix(f"part:{self.stem}:")
            store.meta_set(f"chunks:{self.stem}", sig)
            shutil.rmtree(self.csv_dir, ignore_errors=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        hp = self.csv_dir / "header.json"
        if hp.exists():
            try:
                import json
                self.header = json.loads(hp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.header = None

    # ------------------------------------------------------------------ names
    def part_path(self, p):
        return f"{self.stem}_part{p + 1:03d}.xlsx"

    def piece_path(self, p):
        return self.csv_dir / f"part{p + 1:03d}.csv"

    def bounds(self, p):
        lo = p * self.chunk
        return lo, min(self.n, lo + self.chunk)

    def _marker(self, p):
        return f"part:{self.stem}:{p}"

    def is_finished(self, p):
        if self.store.meta_get(self._marker(p)) != "done" or not self.piece_path(p).exists():
            return False
        x = Path(self.part_path(p))
        return x.exists() or any(x.parent.glob(x.stem + "_*.xlsx"))   # or its timestamped 'file was open' copy

    def mark_row(self, i):
        self.done[i] = True

    def done_in(self, p):
        lo, hi = self.bounds(p)
        return int(self.done[lo:hi].sum())

    def complete(self, p):
        lo, hi = self.bounds(p)
        return (not self.needs_ai) or self.done_in(p) == hi - lo

    # ------------------------------------------------------------------ writing
    def write(self, _df=None):
        """Called by the Autosaver (build() returns None in chunk mode). Returns a short description of what exists."""
        wrote, merged_needed = [], False
        for p in range(self.nparts):
            if self.is_finished(p):
                continue
            nd = self.done_in(p)
            if self.needs_ai and nd == 0:
                continue
            if self._written.get(p) == nd and self.piece_path(p).exists():     # nothing new in this part since the last write
                if self.complete(p):
                    self.store.meta_set(self._marker(p), "done")
                continue
            self._write_part(p)
            wrote.append(p)
            if self.complete(p):
                self.store.meta_set(self._marker(p), "done")
                merged_needed = True
        if wrote and (self.final or merged_needed):
            self.merge_csv()
        elif self.final and not os.path.exists(self.csv_path) and any(self.piece_path(p).exists() for p in range(self.nparts)):
            self.merge_csv()
        return f"{self.stem}_partNNN.xlsx ({self.nparts} parts)"

    def _write_part(self, p):
        lo, hi = self.bounds(p)
        df = self.build_part(lo, hi)
        if self.header is None:
            self.header = [str(c) for c in df.columns]
            import json
            (self.csv_dir / "header.json").write_text(json.dumps(self.header, ensure_ascii=False), encoding="utf-8")
        tgt = self._targets.setdefault(p, report.ExcelTarget(self.part_path(p)))
        path = tgt.write(df)
        self.last_note = tgt.last_note or ""
        piece = self.piece_path(p)
        tmp = str(piece) + ".tmp"
        df.to_csv(tmp, header=False, index=False, encoding="utf-8")
        os.replace(tmp, piece)
        self._written[p] = self.done_in(p)
        self.rows_saved = sum(self.bounds(q)[1] - self.bounds(q)[0] for q in range(self.nparts) if self.piece_path(q).exists())
        del df
        return path

    def merge_csv(self):
        """BOM + header + the pieces in order (pure byte copy, no pandas)."""
        if self.header is None:
            return None
        tmp = self.csv_path + ".tmp"
        head = io.StringIO()
        csv.writer(head, lineterminator="\n").writerow(self.header)
        with open(tmp, "wb") as out:
            out.write(b"\xef\xbb\xbf" + head.getvalue().encode("utf-8"))
            for p in range(self.nparts):
                pp = self.piece_path(p)
                if pp.exists():
                    with open(pp, "rb") as f:
                        shutil.copyfileobj(f, out, 1 << 20)
        try:
            os.replace(tmp, self.csv_path)
        except PermissionError:       # CSV open in Excel on Windows
            alt = self.csv_path[:-4] + "_new.csv"
            os.replace(tmp, alt)
            self.last_note = f"{self.csv_path} is open/locked - merged CSV saved to {alt} instead"
        return self.csv_path

    # ------------------------------------------------------------------ reading back (summary / verdict counts)
    def read_columns(self, wanted):
        """DataFrame with only the wanted columns (those that exist) from all pieces, string dtype, blanks as NaN."""
        if self.header is None:
            return pd.DataFrame(columns=wanted)
        idx = [self.header.index(c) for c in wanted if c in self.header]
        names = [self.header[i] for i in idx]
        frames = []
        for p in range(self.nparts):
            pp = self.piece_path(p)
            if pp.exists() and pp.stat().st_size:
                frames.append(pd.read_csv(pp, header=None, names=self.header, usecols=names, dtype=str,
                                          keep_default_na=False, na_values=[""], encoding="utf-8"))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=names)
