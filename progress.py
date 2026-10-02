"""Live progress for run_qc: output/progress.json (atomic, ~1 s) + a tiny local HTTP server for dashboard/live.html.

Thread-safe. Nothing here can crash a run: file/HTTP failures are swallowed (and noted in `errors`).
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

COUNTER_KEYS = ["form_not_found", "missing_form_link", "missing_photo_link", "photo_is_form", "gps_mismatch",
                "signature_missing", "mismatch", "overwrite", "duplicate_photos", "form_unreadable",
                "not_proforma3", "gps_cluster", "po_id_mismatch", "flooded", "no_crop_in_photo", "crop_mismatch",
                "photo_not_field", "same_location"]
VERDICT_KEYS = ["OK", "Partially OK", "Review", "Manual QC Required"]
SCHEMA_VERSION = 1


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Progress:
    def __init__(self, path="output/progress.json", total=0, title="CLAP AI QC", output_path="", engine="local",
                 html_path=None, write_interval=1.0):
        self.path = str(path)
        self.lock = threading.RLock()
        self.t0 = time.time()
        self.total, self.done, self.done0 = int(total), 0, 0
        self.title, self.engine, self.output_path = title, engine, str(output_path)
        self.status = "Running"
        self.agents: dict[str, dict] = {}
        self.counters = {k: 0 for k in COUNTER_KEYS}
        self.verdicts = {k: 0 for k in VERDICT_KEYS}
        self.events = collections.deque(maxlen=50)
        self.rows = collections.deque(maxlen=3000)      # finished rows (the live table), newest last
        self.row_seq = 0
        self.errors = collections.deque(maxlen=50)
        self.error_count = 0
        self.samples = collections.deque([(self.t0, 0)], maxlen=400)
        self.last_saved, self.rows_saved, self.save_note = None, 0, ""
        self.cost = 0.0
        self.html_path = html_path or str(Path(__file__).parent / "dashboard" / "live.html")
        self.write_interval = write_interval
        self._stop = threading.Event()
        self._thread = None
        self._server = None
        self.url = None
        self.stopped_at = None

    # ------------------------------------------------------------------ updates
    def preload_done(self, n):
        with self.lock:
            self.done = self.done0 = int(n)
            self.samples = collections.deque([(time.time(), self.done)], maxlen=400)

    def set_total(self, n):
        with self.lock:
            self.total = int(n)

    def add_agent(self, name, role):
        with self.lock:
            self.agents[name] = {"id": name, "role": role, "state": "idle", "docket": "", "rows_done": 0,
                                 "avg_sec": 0.0, "errors": 0, "task": "", "_secs": 0.0}

    def agent(self, name, **kw):
        with self.lock:
            a = self.agents.get(name)
            if a is None:
                return
            state = kw.get("state")
            secs = kw.pop("finished_secs", None)
            if secs is not None:
                a["rows_done"] += 1
                a["_secs"] += secs
                a["avg_sec"] = round(a["_secs"] / a["rows_done"], 2)
            if state == "error":
                a["errors"] += 1
            a.update(kw)

    def row_done(self, docket, verdict=None, remark="", counters=(), n=1):
        with self.lock:
            self.done += n
            self.samples.append((time.time(), self.done))
            if verdict in self.verdicts:
                self.verdicts[verdict] += 1
            for c in counters:
                self.counters[c] = self.counters.get(c, 0) + 1
            if remark or verdict:
                self.events.appendleft({"t": datetime.now().strftime("%H:%M:%S"), "docket": str(docket),
                                        "verdict": verdict or "", "remark": str(remark)[:600]})

    def add_row(self, cells: dict):
        """One finished row for the live table (plain strings/numbers only)."""
        with self.lock:
            self.row_seq += 1
            c = {k: ("" if v is None else v) for k, v in cells.items()}
            c["seq"] = self.row_seq
            c["t"] = datetime.now().strftime("%H:%M:%S")
            self.rows.append(c)

    def rows_since(self, since=0, limit=500):
        with self.lock:
            out = [r for r in self.rows if r["seq"] > since][-limit:] if since else list(self.rows)[-limit:]
            return {"last": self.row_seq, "rows": out}

    def error(self, msg):
        with self.lock:
            self.error_count += 1
            self.errors.appendleft({"t": datetime.now().strftime("%H:%M:%S"), "msg": str(msg)[:300]})

    def saved(self, path, rows, note=""):
        with self.lock:
            self.output_path, self.rows_saved, self.save_note = str(path), int(rows), note
            self.last_saved = _now()

    def set_status(self, status):
        with self.lock:
            self.status = status
            if status in ("Done", "Stopped"):
                self.stopped_at = time.time()

    # ------------------------------------------------------------------ snapshot
    def rate(self):
        """Rows per minute over the last ~90 s (overall average while warming up)."""
        with self.lock:
            now = self.stopped_at or time.time()
            cutoff = now - 90
            old = next((s for s in self.samples if s[0] >= cutoff), self.samples[0])
            dt = self.samples[-1][0] - old[0]
            if dt >= 5 and self.samples[-1][1] > old[1]:
                return 60.0 * (self.samples[-1][1] - old[1]) / dt
            el = max(now - self.t0, 1e-6)
            return 60.0 * (self.done - self.done0) / el if self.done > self.done0 else 0.0

    def snapshot(self):
        with self.lock:
            now = self.stopped_at or time.time()
            rate = self.rate()
            remaining = max(self.total - self.done, 0)
            eta = (remaining / rate * 60.0) if rate > 0 and self.status == "Running" else (0 if remaining == 0 else None)
            return {
                "schema": SCHEMA_VERSION,
                "title": self.title,
                "status": self.status,
                "engine": self.engine,
                "started_at": datetime.fromtimestamp(self.t0).strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": _now(),
                "elapsed_sec": round(now - self.t0, 1),
                "total": self.total,
                "done": self.done,
                "percent": round(100.0 * self.done / self.total, 2) if self.total else 100.0,
                "rows_per_min": round(rate, 1),
                "eta_sec": None if eta is None else round(eta),
                "agents": [{k: v for k, v in a.items() if not k.startswith("_")} for a in self.agents.values()],
                "counters": dict(self.counters),
                "verdicts": dict(self.verdicts),
                "events": list(self.events),
                "errors": list(self.errors),
                "error_count": self.error_count,
                "cost": self.cost,
                "output": {"path": self.output_path, "last_saved": self.last_saved, "rows_saved": self.rows_saved,
                           "note": self.save_note},
            }

    # ------------------------------------------------------------------ file writer
    def write_now(self):
        data = json.dumps(self.snapshot(), ensure_ascii=False)
        tmp = self.path + ".tmp"
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp).write_text(data, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:  # e.g. a browser/antivirus holding the file on Windows - skip this tick
            pass
        return data

    def _loop(self):
        while not self._stop.wait(self.write_interval):
            self.write_now()

    def start(self, serve_port=None, host="0.0.0.0"):
        self.write_now()
        self._thread = threading.Thread(target=self._loop, name="progress-writer", daemon=True)
        self._thread.start()
        if serve_port:
            self._serve(serve_port, host)
        return self

    def stop(self, status=None):
        if status:
            self.set_status(status)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.write_now()
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None

    # ------------------------------------------------------------------ HTTP
    def _serve(self, port, host):
        prog = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body: bytes, ctype: str, code=200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                p = self.path.split("?")[0]
                try:
                    if p in ("/", "/live.html", "/index.html"):
                        self._send(Path(prog.html_path).read_bytes(), "text/html; charset=utf-8")
                    elif p == "/favicon.ico":
                        self._send(b"", "image/x-icon", 204)
                    elif p == "/rows.json":
                        q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&") if "=" in x) if "?" in self.path else {}
                        self._send(json.dumps(prog.rows_since(int(q.get("since", 0) or 0)), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                    elif p == "/rows.csv":
                        import csv, io
                        rows = prog.rows_since(0, 3000)["rows"]
                        buf = io.StringIO(); w = csv.writer(buf)
                        cols = ["docket", "farmer", "village", "surveyor", "verdict", "form_no", "po_id", "area_form", "loss_form", "area_app", "loss_app", "match",
                                "farmer_sig", "company_sig", "worker_sig", "photo_is_form", "person", "flags", "remark"]
                        w.writerow(cols)
                        for r in rows:
                            w.writerow([r.get(c, "") for c in cols])
                        self._send(("\ufeff" + buf.getvalue()).encode("utf-8"), "text/csv; charset=utf-8")
                    elif p == "/progress.json":
                        self._send(json.dumps(prog.snapshot(), ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                    else:
                        self._send(b"not found", "text/plain", 404)
                except (BrokenPipeError, ConnectionError):
                    pass
                except Exception as e:  # never let a request kill anything
                    try:
                        self._send(str(e).encode(), "text/plain", 500)
                    except Exception:
                        pass

        for p in range(int(port), int(port) + 10):
            try:
                srv = ThreadingHTTPServer((host, p), H)
            except OSError:
                continue
            srv.daemon_threads = True
            threading.Thread(target=srv.serve_forever, name="progress-http", daemon=True).start()
            self._server = srv
            self.port = p
            self.url = f"http://localhost:{p}/"
            return
        self.error(f"could not start the live dashboard server on ports {port}-{int(port) + 9}")


def lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None
