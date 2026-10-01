#!/usr/bin/env python3
"""Merge the CSV results of several batches (PCs / nights) into ONE UTF-8-with-BOM CSV that Excel opens (Hindi text intact).

    python tools/merge_qc.py batch_000.csv batch_001.csv batch_002.csv --output all_rows.csv
    python tools/merge_qc.py "results/*.csv" --output all_rows.csv        (quotes: the program expands the pattern)

All files must come from run_qc.py of the same version (identical header). Files are copied byte by byte, so 160,000 rows
(~400 MB) merge in seconds without using memory. Rows are NOT de-duplicated; `--check-dupes` lists repeated dockets.
Excel itself cannot hold all rows comfortably - open the CSV in Excel/Power BI/pandas, or keep the per-batch part files.
"""
import argparse
import csv
import glob
import io
import shutil
import sys
from pathlib import Path


def read_header(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return next(csv.reader(f), [])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="CSV files or glob patterns, in the order they should appear")
    ap.add_argument("--output", required=True)
    ap.add_argument("--check-dupes", action="store_true", help="report dockets that appear more than once")
    a = ap.parse_args(argv)
    files = []
    for pat in a.inputs:
        hit = sorted(glob.glob(pat)) or ([pat] if Path(pat).exists() else [])
        files += hit
    files = [f for f in files if Path(f).resolve() != Path(a.output).resolve()]
    if not files:
        sys.exit("no input files found")
    head = read_header(files[0])
    for f in files[1:]:
        if read_header(f) != head:
            sys.exit(f"header of {f} differs from {files[0]} - were they made by the same version of run_qc.py?")
    tmp = a.output + ".tmp"
    with open(tmp, "wb") as out:
        out.write(b"\xef\xbb\xbf")
        for k, f in enumerate(files):
            with open(f, "rb") as src:
                first = src.read(3)
                if first != b"\xef\xbb\xbf":
                    src.seek(0)
                if k:   # drop the header line of every file but the first (header never contains a quoted newline)
                    src.readline()
                shutil.copyfileobj(src, out, 1 << 20)
    Path(tmp).replace(a.output)
    print(f"Merged {len(files)} file(s) -> {a.output}")
    if a.check_dupes:
        import pandas as pd
        col = "docket_id" if "docket_id" in head else head[0]
        d = pd.read_csv(a.output, usecols=[col], dtype=str, encoding="utf-8-sig")[col]
        dup = d[d.duplicated(keep=False)].unique()
        print(f"rows: {len(d):,}  repeated {col}s: {len(dup):,}" + (f"  e.g. {list(dup[:5])}" if len(dup) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
