"""Crash-safe, compact result store for big runs (replaces the one-big-JSON checkpoint for the offline engine).

SQLite in WAL mode: every finished row is committed immediately (so a Ctrl-C, kill -9 or power loss loses at most the rows that
were in flight), memory use is flat (only the set of finished keys is kept, never the per-row result dicts), and a chunk of rows can
be read back by key without loading the rest. Behaves like a dict for the few operations the engine needs.

Old checkpoints (`checkpoint.json` = {"results": {...}, "cost": x}) are imported once, automatically, by `open_store(..., legacy_json=...)`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path


def store_path_for(checkpoint: str) -> str:
    """`output/checkpoint.json` -> `output/checkpoint.sqlite` (an explicit .sqlite/.db path is used as is)."""
    p = str(checkpoint)
    if p.lower().endswith((".sqlite", ".db", ".sqlite3")):
        return p
    if p.lower().endswith(".json"):
        return p[:-5] + ".sqlite"
    return p + ".sqlite"


class ResultStore:
    def __init__(self, path, fresh=False):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            for suf in ("", "-wal", "-shm", "-journal"):
                try:
                    os.remove(self.path + suf)
                except OSError:
                    pass
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, timeout=60, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute("CREATE TABLE IF NOT EXISTS results(key TEXT PRIMARY KEY, done INTEGER NOT NULL, js TEXT NOT NULL)")
        self._db.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
        self._done = {k for (k,) in self._db.execute("SELECT key FROM results WHERE done=1")}

    # ------------------------------------------------------------------ dict-like
    def __setitem__(self, key, res):
        js = json.dumps(res, default=str, ensure_ascii=False, separators=(",", ":"))
        done = 1 if isinstance(res, dict) and "verdict" in res else 0
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO results(key, done, js) VALUES(?,?,?)", (str(key), done, js))
            (self._done.add if done else self._done.discard)(str(key))

    def get(self, key, default=None):
        with self._lock:
            r = self._db.execute("SELECT js FROM results WHERE key=?", (str(key),)).fetchone()
        return json.loads(r[0]) if r else default

    def __getitem__(self, key):
        r = self.get(key, KeyError)
        if r is KeyError:
            raise KeyError(key)
        return r

    def __contains__(self, key):
        with self._lock:
            return self._db.execute("SELECT 1 FROM results WHERE key=?", (str(key),)).fetchone() is not None

    def __len__(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM results").fetchone()[0]

    def keys(self):
        with self._lock:
            return [k for (k,) in self._db.execute("SELECT key FROM results")]

    def items(self):
        for k in self.keys():
            yield k, self.get(k)

    def setdefault(self, key, default):
        cur = self.get(key)
        if cur is None:
            self[key] = default
            return default
        return cur

    def get_many(self, keys):
        """list of result dicts (or None) aligned to `keys`; reads in batches."""
        out, keys = {}, [str(k) for k in keys]
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = "SELECT key, js FROM results WHERE key IN (%s)" % ",".join("?" * len(part))
                for k, js in self._db.execute(q, part):
                    out[k] = js
        return [json.loads(out[k]) if k in out else None for k in keys]

    # ------------------------------------------------------------------ helpers
    def done_keys(self):
        """keys of rows with a final verdict (a copy; cheap, kept in memory as a set of short strings)."""
        with self._lock:
            return set(self._done)

    def is_done(self, key):
        return str(key) in self._done

    def meta_get(self, k, default=None):
        with self._lock:
            r = self._db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def meta_set(self, k, v):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO meta(k, v) VALUES(?,?)", (k, str(v)))

    def meta_delete_prefix(self, prefix):
        with self._lock:
            self._db.execute("DELETE FROM meta WHERE k LIKE ?", (prefix + "%",))

    def import_legacy_json(self, json_path):
        """One-time import of an old checkpoint.json (results + cost). Safe to call repeatedly (skipped when already imported)."""
        p = Path(json_path)
        try:
            st = p.stat()
        except OSError:
            return 0
        sig = f"{st.st_size}:{int(st.st_mtime)}"
        if self.meta_get("legacy_json") == sig:
            return 0
        try:
            ck = json.loads(p.read_text())
            res = ck["results"]
            if not isinstance(res, dict):
                raise ValueError("bad checkpoint")
        except (OSError, ValueError, TypeError, KeyError):
            return 0
        n = 0
        with self._lock:
            self._db.execute("BEGIN")
            try:
                for k, v in res.items():
                    if k in self._done:       # never overwrite a newer result
                        continue
                    self[k] = v
                    n += 1
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        self.meta_set("legacy_json", sig)
        try:
            self.meta_set("cost", float(ck.get("cost") or 0.0))
        except (TypeError, ValueError):
            pass
        return n

    def close(self):
        with self._lock:
            try:
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._db.close()
            except sqlite3.Error:
                pass


def open_store(checkpoint, resume, readonly_legacy=True):
    """Open the sqlite store next to `checkpoint`. A fresh run wipes it; --resume keeps it and imports an old JSON checkpoint once."""
    sp = store_path_for(checkpoint)
    st = ResultStore(sp, fresh=not resume)
    if resume and str(checkpoint).lower().endswith(".json") and os.path.exists(checkpoint):
        n = st.import_legacy_json(checkpoint)
        if n:
            print(f"Imported {n:,} rows from the old JSON checkpoint {checkpoint} into {sp}")
    return st
