"""Offline AI-QC pipeline: downloader threads feed a pool of worker PROCESSES ("agents").

    downloader slots (asyncio, I/O-bound)  ->  form readers / photo analysts (processes, CPU-bound)
                                            ->  row finalizer (remarks.evaluate) -> checkpoint results

* Every agent is a separate spawn-started process (Windows-safe) with OMP_THREAD_LIMIT=1; models are loaded once per agent.
* Per-agent duplex pipes (not shared queues), so a killed/crashed worker can never corrupt another agent's channel.
* A dead or hung worker is detected, respawned, and its row part is retried once; after that the row gets an 'error' part
  and the run carries on. One bad row never kills the run.
* --inline / inline=True runs the same agents as threads inside this process (used by tests and for debugging).

The old async adapters process_pdf / process_photo are kept at the bottom for the --engine claude/legacy code paths.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import importlib
import importlib.util
import multiprocessing as mp
import os
import re
import threading
import time
from multiprocessing import connection as mpc
from pathlib import Path

import config as C
import remarks
from common import Unavailable, fetch, num

FORM_DIRS = {"form", "forms", "pdf", "pdfs", "signed", "signed_copy"}
GUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
ROW_FIELDS = ("docket_id", "latitude", "longitude", "affected_area_pct", "crop_loss_pct", "total_damage_pct",
              "survey_start_date", "survey_end_date", "crop_name", "pdf_url", "media_urls", "state", "district")


# ============================================================================ sanitising / engines (worker side)
def sanitize(o, depth=0):
    """Make an engine result JSON-safe and small (numpy -> python, drop arrays/images/bytes)."""
    if depth > 4:
        return None
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return None if o != o else o
    if hasattr(o, "item") and getattr(o, "shape", None) == ():
        return sanitize(o.item(), depth)
    if isinstance(o, dict):
        return {str(k): sanitize(v, depth + 1) for k, v in o.items() if not str(k).startswith("_img")}
    if isinstance(o, (list, tuple, set)):
        return [sanitize(v, depth + 1) for v in list(o)[:60]]
    if isinstance(o, Path):
        return str(o)
    return str(o)[:200] if not hasattr(o, "shape") and not hasattr(o, "size") else None


def _fallback_form(path, docket=None):
    """Used until form_reader.py exists: the old generic Tesseract reader mapped to the read_form keys."""
    import local_ocr
    r = local_ocr.extract(path)
    return {"is_proforma3": None, "quality": None, "form_no": None, "po_id": None, "po_id_matches": None,
            "form_area": r.get("form_area"), "form_loss": r.get("form_loss"), "row_area": None, "row_loss": None,
            "total_row_blank": None, "loss_date": r.get("survey_date"),
            "farmer_signed": None, "company_signed": None, "worker_signed": None, "officer_signed": None,
            "officer_stamp_only": False, "overwrite_suspected": False,
            # the generic reader cannot read this handwriting/layout reliably: never report it as confident
            "confidence": min(float(r.get("pdf_confidence") or 0.0), 0.5), "field_conf": {},
            "notes": ["generic OCR fallback (form_reader not installed)"], "engine": "local-ocr"}


def load_engines(spec=None):
    """(read_form, analyse, label). spec='module' provides both (tests); else form_reader/photo_local with fallbacks."""
    if spec:
        m = importlib.import_module(spec)
        return m.read_form, m.analyse, spec
    try:
        from form_reader import read_form
        fl = "form_reader"
    except ImportError:
        read_form, fl = _fallback_form, "generic-ocr"
    import photo_local
    return read_form, photo_local.analyse, f"{fl}+photo_local"


def _accepts_form_image(fn):
    import inspect
    try:
        ps = inspect.signature(fn).parameters
        return "form_image" in ps or any(p.kind == p.VAR_KEYWORD for p in ps.values())
    except (TypeError, ValueError):
        return False


def _run_task(task, form_fn, photo_fn, photo_takes_form):
    if task["kind"] == "form":
        res = form_fn(task["paths"][0], docket=task.get("docket"))
    else:
        row = task["row"]
        fi = None
        if photo_takes_form and task.get("form_path") and not str(task["form_path"]).lower().endswith(".pdf"):
            try:
                from PIL import Image
                fi = Image.open(task["form_path"])
                fi.thumbnail((1000, 1000))
                fi = fi.convert("RGB")
            except Exception:
                fi = None
        res = photo_fn(task["paths"], row, form_image=fi) if photo_takes_form else photo_fn(task["paths"], row)
    res = sanitize(res if isinstance(res, dict) else {"value": res})
    res.setdefault("_state", "ok")
    return res


def _worker_main(name, role, conn, engine_spec):
    try:  # Ctrl-C is handled by the coordinator; agents must not die with a traceback
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        pass
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        import cv2
        cv2.setNumThreads(1)
    except Exception:
        pass
    try:
        form_fn, photo_fn, label = load_engines(engine_spec)
        takes = _accepts_form_image(photo_fn)
        if not engine_spec:
            for mod in ("form_reader", "photo_local", "digits"):
                try:
                    w = getattr(importlib.import_module(mod), "warmup", None)
                    if callable(w):
                        w()
                except Exception:
                    pass
        conn.send(("ready", os.getpid(), label))
    except BaseException as e:  # noqa
        try:
            conn.send(("fatal", f"{type(e).__name__}: {e}"[:300]))
        finally:
            return
    while True:
        try:
            task = conn.recv()
        except (EOFError, OSError):
            return
        if task is None:
            return
        t = time.time()
        try:
            res = _run_task(task, form_fn, photo_fn, takes)
        except Exception as e:  # one bad row must never kill the agent
            res = {"_state": "error", "_error": f"{type(e).__name__}: {e}"[:200]}
        try:
            conn.send(("done", task["id"], res, time.time() - t))
        except (OSError, ValueError):
            return


# ============================================================================ media classification
def classify_local(files, docket, pdf_url=None):
    """Split a docket's local files into (form_paths, photo_paths).

    form = any PDF; an image whose name is exactly the docket id, lives in a forms/pdf folder, or whose GUID is in the
    row's signed-form URL. Everything else is a photo.
    """
    guids = {g.lower() for g in GUID.findall(str(pdf_url or ""))}
    forms, photos = [], []
    for p in files.get("pdfs", []):
        forms.append(p)
    for p in files.get("images", []):
        p = Path(p)
        if p.stem == str(docket) or p.parent.name.lower() in FORM_DIRS or any(g.lower() in guids for g in GUID.findall(p.name)):
            forms.append(p)
        else:
            photos.append(p)
    forms.sort(key=lambda p: (p.suffix.lower() != ".pdf", str(p)))
    return forms, sorted(photos, key=str)


def _sniff_ext(path):
    try:
        b = Path(path).read_bytes()[:12]
    except OSError:
        return ""
    if b[:4] == b"%PDF":
        return ".pdf"
    if b[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if b[:4] == b"RIFF":
        return ".webp"
    return ""


def _with_ext(path):
    """cache files are hash-named; give them a real extension (engines dispatch on it)."""
    path = Path(path)
    if path.suffix:
        return path
    ext = _sniff_ext(path)
    if not ext:
        return path
    dst = path.with_name(path.name + ext)
    if not dst.exists():
        try:
            os.link(path, dst)
        except OSError:
            try:
                import shutil
                shutil.copyfile(path, dst)
            except OSError:
                return path
    return dst


# ============================================================================ the runner
class _Worker:
    def __init__(self, name, role, label):
        self.name, self.role, self.label = name, role, label
        self.proc = self.conn = None
        self.state = "starting"
        self.task = None
        self.t_dispatch = 0.0
        self.respawns = 0


class LocalRunner:
    def __init__(self, rows, keys, todo, ck, kinds, local_map, ctx, *, agents=4, downloaders=None, rate=5.0,
                 engine_module=None, inline=False, progress=None, task_timeout=180.0, pause_file=None,
                 cache_dirs=None, max_inflight=None, on_row=None, discard_media=False):
        self.rows, self.keys, self.todo, self.ck = rows, keys, list(todo), ck
        self.kinds = set(kinds)
        self.local_map = local_map or {}
        self.ctx = ctx                      # dict of lists aligned to rows: gps, dflags, risk
        self.n_agents = max(1, int(agents))
        self.n_dl = int(downloaders) if downloaders else max(2, min(8, self.n_agents))
        self.rate = float(rate)
        self.engine_spec, self.inline = engine_module, inline
        self.prog = progress
        self.task_timeout = task_timeout
        self.pause_file = pause_file
        self.cache = cache_dirs or {"form": C.PDF_CACHE_DIR, "photo": C.PHOTO_CACHE_DIR}
        self.max_inflight = max_inflight or max(8, self.n_agents * 4)
        self.on_row = on_row
        self.discard_media = bool(discard_media)
        self._refs = collections.Counter()   # downloaded cache file -> rows still using it (discard mode)
        self.discarded = 0
        self.lock = threading.RLock()
        self.state: dict[int, dict] = {}
        self.pending = {"form": collections.deque(), "photo": collections.deque()}
        self.workers: list[_Worker] = []
        self.stop_evt = threading.Event()
        self.finalized = 0
        self.feeder_done = False
        self._task_id = 0
        self.mpctx = mp.get_context("spawn")
        self.engine_label = engine_module or "?"
        self.errors = 0
        self._loop = None
        self._feed_task = None
        self._sem = None
        self._next_slot = 0.0

    # ------------------------------------------------------------------ public
    def run(self):
        if not self.todo:
            return
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        if self.discard_media:
            self._purge_stale_cache()
        self._make_workers()
        feeder = threading.Thread(target=self._feeder_main, name="downloaders", daemon=True)
        feeder.start()
        try:
            self._coordinate()
        finally:
            self.stop_evt.set()
            self._shutdown(feeder)
            if self.discard_media:  # rows that were in flight when we stopped: do not leave their downloads behind
                with self.lock:
                    left = list(self.state.values())
                for st in left:
                    self._release(st)

    def stop(self):
        self.stop_evt.set()

    # ------------------------------------------------------------------ --discard-media (bounded cache)
    def _purge_stale_cache(self, max_age=3600):
        """Only half-written download temp files (`*.part<pid>`) of an earlier killed run are removed - never other cached files."""
        now = time.time()
        for d in self.cache.values():
            try:
                for p in Path(d).glob("*.part*"):
                    if p.is_file() and now - p.stat().st_mtime > max_age:
                        p.unlink()
            except OSError:
                pass

    def _track(self, st, *paths):
        """Remember downloaded cache files of this row (never called for --local-media files)."""
        if not self.discard_media:
            return
        with self.lock:
            for p in {str(x) for x in paths}:
                if p not in st.setdefault("dl", set()):
                    st["dl"].add(p)
                    self._refs[p] += 1

    def _release(self, st):
        """Delete this row's downloaded files once nothing else uses them. Only files inside the cache dirs are ever removed."""
        roots = [Path(d).resolve() for d in self.cache.values()]
        with self.lock:
            paths = list(st.get("dl", ()))
            st["dl"] = set()
            gone = []
            for p in paths:
                self._refs[p] -= 1
                if self._refs[p] <= 0:
                    del self._refs[p]
                    gone.append(p)
        import glob
        sidecar_dirs = {str(r) for r in roots} | {str(Path(__file__).resolve().parent / C.PHOTO_CACHE_DIR)}
        for p in gone:
            try:
                rp = Path(p).resolve()
                if any(r in rp.parents for r in roots):
                    rp.unlink()
                    self.discarded += 1
                    # the photo analyser keeps a feature cache `<photo name>.<size>.<mtime>.<ver>.pkl` (~45 KB each): drop it with the photo
                    for d in sidecar_dirs:
                        for side in glob.glob(os.path.join(glob.escape(d), glob.escape(rp.name) + ".*.pkl")):
                            try:
                                os.remove(side)
                            except OSError:
                                pass
            except OSError:
                pass

    # ------------------------------------------------------------------ workers
    def _roles(self):
        n = self.n_agents
        if "photo" not in self.kinds:
            return ["form"] * n
        if "form" not in self.kinds:
            return ["photo"] * n
        nf = max(1, round(n * 0.6)) if n > 1 else 1
        return ["form"] * nf + ["photo"] * (n - nf)

    def _make_workers(self):
        counts = collections.Counter()
        for role in self._roles():
            counts[role] += 1
            label = ("Form reader" if role == "form" else "Photo analyst") + f" #{counts[role]}"
            w = _Worker(f"{role}{counts[role]}", role, label)
            self.workers.append(w)
            if self.prog:
                self.prog.add_agent(w.name, label)
            self._spawn(w)
        if self.prog:
            for k in range(1, self.n_dl + 1):
                self.prog.add_agent(f"dl{k}", f"Downloader #{k}")

    def _spawn(self, w):
        parent, child = self.mpctx.Pipe(duplex=True)
        if self.inline:
            w.proc = threading.Thread(target=_worker_main, args=(w.name, w.role, child, self.engine_spec), daemon=True)
        else:
            w.proc = self.mpctx.Process(target=_worker_main, args=(w.name, w.role, child, self.engine_spec), daemon=True)
        w.proc.start()
        if not self.inline:
            child.close()
        w.conn, w.state, w.task = parent, "starting", None
        if self.prog:
            self.prog.agent(w.name, state="starting", docket="", task="loading models")

    def _kill(self, w):
        try:
            if not self.inline and w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(2)
        except Exception:
            pass
        try:
            w.conn.close()
        except Exception:
            pass

    def _shutdown(self, feeder):
        if self._loop and self._feed_task:
            try:
                self._loop.call_soon_threadsafe(self._feed_task.cancel)
            except RuntimeError:
                pass
        for w in self.workers:
            try:
                w.conn.send(None)
            except Exception:
                pass
        t_end = time.time() + 2.0
        for w in self.workers:
            try:
                w.proc.join(max(0.0, t_end - time.time()))
            except Exception:
                pass
        for w in self.workers:
            self._kill(w)
        feeder.join(timeout=3)

    # ------------------------------------------------------------------ coordinator
    def _coordinate(self):
        while not self.stop_evt.is_set():
            if self.pause_file and os.path.exists(self.pause_file):
                if self.prog:
                    self.prog.set_status("Paused")
                time.sleep(0.5)
                continue
            if self.prog and self.prog.status == "Paused":
                self.prog.set_status("Running")
            with self.lock:
                finished = self.feeder_done and (self.finalized >= len(self.todo) or (
                    not self.state and not any(self.pending.values()) and not any(w.state == "busy" for w in self.workers)))
            if finished:
                return
            self._dispatch()
            conns = [w.conn for w in self.workers if w.state in ("starting", "idle", "busy")]
            try:
                ready = mpc.wait(conns, timeout=0.2) if conns else []
                if not conns:
                    time.sleep(0.2)
            except (OSError, ValueError):
                ready = []
            by_conn = {w.conn: w for w in self.workers}
            for c in ready:
                w = by_conn.get(c)
                if w is None:
                    continue
                try:
                    msg = c.recv()
                except (EOFError, OSError):
                    self._on_death(w, "worker exited")
                    continue
                self._on_msg(w, msg)
            self._health()
            self._flush_if_no_workers()

    def _dispatch(self):
        for w in self.workers:
            if w.state != "idle":
                continue
            with self.lock:
                task = None
                for kind in (w.role, "photo" if w.role == "form" else "form"):
                    if self.pending[kind]:
                        task = self.pending[kind].popleft()
                        break
            if task is None:
                continue
            try:
                w.conn.send(task)
            except (OSError, ValueError):
                with self.lock:
                    self.pending[task["kind"]].appendleft(task)
                self._on_death(w, "pipe closed")
                continue
            w.state, w.task, w.t_dispatch = "busy", task, time.time()
            if self.prog:
                helping = "" if task["kind"] == w.role else " (helping)"
                self.prog.agent(w.name, state="busy", docket=str(task["docket"]), task=task["kind"] + helping)

    def _on_msg(self, w, msg):
        kind = msg[0]
        if kind == "ready":
            w.state = "idle"
            self.engine_label = msg[2] if len(msg) > 2 else self.engine_label
            if self.prog:
                self.prog.agent(w.name, state="idle", task="")
        elif kind == "fatal":
            w.respawns += 99  # the engines cannot be loaded; do not respawn-loop
            self._on_death(w, "engine load failed: " + str(msg[1]))
        elif kind == "done":
            _, tid, res, secs = msg
            task, w.task, w.state = w.task, None, "idle"
            if self.prog:
                ok = res.get("_state") != "error"
                self.prog.agent(w.name, state="idle" if ok else "error", docket="", task="", finished_secs=secs)
            if task is not None and task["id"] == tid:
                self._part_done(task, res, secs)

    def _on_death(self, w, why):
        task, w.task = w.task, None
        w.state = "dead"
        self._kill(w)
        self.errors += 1
        if self.prog:
            self.prog.agent(w.name, state="error", docket="", task=why[:60])
            self.prog.error(f"{w.label}: {why}" + (f" while on {task['docket']}" if task else ""))
        if task is not None:
            if task["attempt"] == 0:
                task["attempt"] = 1
                with self.lock:
                    self.pending[task["kind"]].appendleft(task)
            else:
                self._part_done(task, {"_state": "error", "_error": f"worker failed twice: {why}"[:200]}, 0.0)
        if w.respawns < 5:
            w.respawns += 1
            self._spawn(w)

    def _health(self):
        now = time.time()
        for w in self.workers:
            if w.state in ("starting", "idle", "busy") and not self.inline and not w.proc.is_alive():
                # drain anything it managed to send before dying
                try:
                    while w.conn.poll():
                        self._on_msg(w, w.conn.recv())
                except (EOFError, OSError):
                    pass
                if w.state != "dead":
                    self._on_death(w, "worker process died")
            elif w.state == "busy" and now - w.t_dispatch > self.task_timeout:
                self._on_death(w, f"task timed out after {self.task_timeout:.0f}s")

    def _flush_if_no_workers(self):
        if any(w.state != "dead" for w in self.workers):
            return
        with self.lock:
            tasks = [t for k in self.pending for t in self.pending[k]]
            for k in self.pending:
                self.pending[k].clear()
        for t in tasks:
            self._part_done(t, {"_state": "error", "_error": "no working agents left"}, 0.0)
        if self.feeder_done and not tasks:
            pass

    # ------------------------------------------------------------------ row state
    def _add_task(self, i, kind, paths, extra=None):
        with self.lock:
            self._task_id += 1
            task = {"id": self._task_id, "i": i, "kind": kind, "docket": self.rows[i].get("docket_id"), "paths": [str(p) for p in paths],
                    "row": {k: self.rows[i].get(k) for k in ROW_FIELDS}, "attempt": 0}
            if extra:
                task.update(extra)
            self.pending[kind].append(task)

    def _part_done(self, task, res, secs):
        with self.lock:
            st = self.state[task["i"]]
            st["parts"][task["kind"]] = res
            st["secs"] += secs
            st["need"] -= 1
            complete = st["need"] <= 0 and st["prepared"]
        if complete:
            self._finalize(task["i"])

    def _prepared(self, i):
        with self.lock:
            st = self.state[i]
            st["prepared"] = True
            complete = st["need"] <= 0
        if complete:
            self._finalize(i)

    @staticmethod
    def _live_cells(row, form, photos, ev):
        def yn(v):
            return "" if v is None else ("Yes" if v else "No")
        f = form if isinstance(form, dict) and form.get("_state") not in ("error", "not_found", "no_link", "skipped") else {}
        p = photos if isinstance(photos, dict) else {}
        fa, fl = ev.get("form_area"), ev.get("form_loss")
        return {"docket": row.get("docket_id"), "farmer": row.get("farmer_name"), "village": row.get("village"), "surveyor": row.get("surveyor_name"),
                "verdict": ev.get("verdict"), "confidence": ev.get("confidence"),
                "form_no": f.get("form_no") if f.get("form_no_conf", 1) >= 0.9 else "",
                "area_form": fa, "loss_form": fl, "area_app": row.get("affected_area_pct"), "loss_app": row.get("crop_loss_pct"),
                "match": ev.get("match"), "farmer_sig": yn(f.get("farmer_signed")), "company_sig": yn(f.get("company_signed")),
                "worker_sig": yn(f.get("worker_signed")), "photo_is_form": "Yes" if p.get("scene_type") == "paper form" else "",
                "person": "", "flags": "; ".join(ev.get("flags") or []), "remark": ev.get("remark")}

    def _finalize(self, i):
        with self.lock:
            st = self.state[i]
            if st.get("final"):
                return
            st["final"] = True
        row = self.rows[i]
        form = st["parts"].get("form") if "form" in self.kinds else {"_state": "skipped"}
        photos = st["parts"].get("photo") if "photo" in self.kinds else {"_state": "skipped"}
        if isinstance(photos, dict) and photos.get("_state") != "skipped":
            photos["_n_photos"] = st.get("n_photos", 0)
        gps = self.ctx["gps"][i] if self.ctx.get("gps") else {}
        dfl = self.ctx["dflags"][i] if self.ctx.get("dflags") else ""
        risk = self.ctx["risk"][i] if self.ctx.get("risk") else {}
        try:
            ev = remarks.evaluate(row, form, photos, gps, dfl, risk)
        except Exception as e:  # remark building must never lose the row
            ev = {"remark": f"Remark could not be built ({type(e).__name__}: {e})", "verdict": "Manual-check", "confidence": "Low",
                  "flags": ["Internal error"], "counters": [], "match": "NA", "form_area": None, "form_loss": None}
            if self.prog:
                self.prog.error(f"remark build failed for {row.get('docket_id')}: {e}")
        res = {"engine": "local", "form": form, "photo": photos,
               "pdf_status": _status(form), "photo_status": _status(photos),
               "verdict": ev["verdict"], "confidence": ev["confidence"], "flags": ev["flags"], "remark": ev["remark"], "remark_detail": ev.get("remark_detail", ""),
               "match": ev["match"], "form_area": ev["form_area"], "form_loss": ev["form_loss"],
               "n_photos": st.get("n_photos", 0), "secs": round(st["secs"], 2),
               "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        with self.lock:
            self.ck["results"][self.keys[i]] = res
            self.finalized += 1
            self.state.pop(i, None)
        if self.discard_media:
            self._release(st)
        if self.prog:
            self.prog.row_done(row.get("docket_id"), ev["verdict"], ev["remark"], ev["counters"])
            try:
                self.prog.add_row(self._live_cells(row, form, photos, ev))
            except Exception:       # the live table is cosmetic; never lose a row over it
                pass
        if self.on_row:
            try:
                self.on_row(i, res)
            except Exception:
                pass
        if self._loop and self._sem:
            try:
                self._loop.call_soon_threadsafe(self._sem.release)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------ downloader side (asyncio thread)
    def _feeder_main(self):
        try:
            asyncio.run(self._feed())
        except BaseException as e:  # noqa
            if self.prog and not isinstance(e, asyncio.CancelledError):
                self.prog.error(f"downloader crashed: {type(e).__name__}: {e}")
        finally:
            self.feeder_done = True

    async def _gate(self):
        """Global polite rate limit on real network requests."""
        if self.rate <= 0:
            return
        now = time.monotonic()
        slot = max(now, self._next_slot)
        self._next_slot = slot + 1.0 / self.rate
        if slot > now:
            await asyncio.sleep(slot - now)

    async def _feed(self):
        import httpx
        self._loop = asyncio.get_running_loop()
        self._feed_task = asyncio.current_task()
        self._sem = asyncio.Semaphore(self.max_inflight)
        q: asyncio.Queue = asyncio.Queue(maxsize=self.n_dl * 2)
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=self.n_dl * 2),
                                     headers={"User-Agent": "Mozilla/5.0 (compatible; CLAP-QC-offline)"}) as http:
            async def producer():
                for i in self.todo:
                    if self.stop_evt.is_set():
                        break
                    await self._sem.acquire()
                    await q.put(i)
                for _ in range(self.n_dl):
                    await q.put(None)

            async def consumer(k):
                name = f"dl{k}"
                while True:
                    i = await q.get()
                    if i is None or self.stop_evt.is_set():
                        if self.prog:
                            self.prog.agent(name, state="idle", docket="", task="")
                        return
                    t = time.time()
                    if self.prog:
                        self.prog.agent(name, state="busy", docket=str(self.rows[i].get("docket_id")), task="fetching media")
                    try:
                        await self._prepare(i, http)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        self._prepare_failed(i, e)
                    if self.prog:
                        self.prog.agent(name, state="idle", docket="", task="", finished_secs=time.time() - t)

            await asyncio.gather(producer(), *(consumer(k) for k in range(1, self.n_dl + 1)))

    def _prepare_failed(self, i, e):
        with self.lock:
            st = self.state.setdefault(i, {"need": 0, "parts": {}, "secs": 0.0, "prepared": False})
        for kind in self.kinds:
            with self.lock:
                if kind not in st["parts"] and st["need"] <= 0:
                    st["parts"][kind] = {"_state": "error", "_error": f"{type(e).__name__}: {e}"[:160]}
        self._prepared(i)

    async def _get(self, http, url, kind, st=None):
        cache = self.cache[kind]
        p = Path(cache) / hashlib.sha1(url.encode()).hexdigest()
        if not (p.exists() and p.stat().st_size):
            await self._gate()
        timeout = C.PDF_TIMEOUT_SEC if kind == "form" else C.PHOTO_TIMEOUT_SEC
        raw = await fetch(http, url, cache, timeout)
        got = _with_ext(raw)
        if st is not None:
            self._track(st, raw, got)
        return got

    async def _prepare(self, i, http):
        from photo_qc import split_urls
        row = self.rows[i]
        docket = str(row.get("docket_id")).strip()
        with self.lock:
            self.state[i] = st = {"need": 0, "parts": {}, "secs": 0.0, "prepared": False, "n_photos": 0}
        forms, photos = classify_local(self.local_map.get(docket, {}), docket, row.get("pdf_url"))
        tasks = []
        if "form" in self.kinds:
            url = str(row.get("pdf_url") or "").replace("_x000D_", "").strip()
            has_url = url.startswith("http")
            fpath, err = (forms[0] if forms else None), None
            if fpath is None and has_url:
                try:
                    fpath = await self._get(http, url, "form", st)
                except Unavailable as e:
                    err = str(e)
                except Exception as e:
                    err = f"{type(e).__name__}"
            if fpath is not None:
                tasks.append(("form", [fpath], None))
                st["form_path"] = str(fpath)
            elif has_url:
                st["parts"]["form"] = {"_state": "not_found", "_error": err or "unavailable"}
            else:
                st["parts"]["form"] = {"_state": "no_link"}
        if "photo" in self.kinds:
            paths = list(photos)[:C.MAX_PHOTOS_PER_ROW]
            urls = split_urls(row.get("media_urls"))
            err = None
            for u in urls[:max(0, C.MAX_PHOTOS_PER_ROW - len(paths))] if not paths else []:
                try:
                    paths.append(await self._get(http, u, "photo", st))
                except Unavailable as e:
                    err = str(e)
                except Exception as e:
                    err = type(e).__name__
            if paths:
                st["n_photos"] = len(paths)
                tasks.append(("photo", paths, {"form_path": st.get("form_path")}))
            elif urls:
                st["parts"]["photo"] = {"_state": "not_found", "_error": err or "unavailable"}
            else:
                st["parts"]["photo"] = {"_state": "no_link"}
        with self.lock:
            st["need"] = len(tasks)
        for kind, paths, extra in tasks:
            self._add_task(i, kind, paths, extra)
        self._prepared(i)


def _status(part):
    s = (part or {}).get("_state")
    return "OK" if s == "ok" else ("Error" if s == "error" else ("Unavailable" if s in ("not_found", "no_link") else ("Skipped" if s == "skipped" else None)))


# ============================================================================ legacy async adapters (old code paths)
async def process_pdf(row, client, http, tracker):
    import local_ocr
    local = row.get("_local_pdfs") or []
    url = str(row.get("pdf_url") or "").replace("_x000D_", "").strip()
    if not local and not url.startswith("http"):
        return {"pdf_status": "Unavailable", "pdf_error": "no url"}
    try:
        path = local[0] if local else await fetch(http, url, C.PDF_CACHE_DIR, C.PDF_TIMEOUT_SEC)
        return await asyncio.to_thread(local_ocr.extract, path, num(row.get("affected_area_pct")), num(row.get("crop_loss_pct")))
    except Unavailable as e:
        return {"pdf_status": "Unavailable", "pdf_error": str(e)}
    except Exception as e:
        return {"pdf_status": "Error", "pdf_error": f"{type(e).__name__}: {e}"[:200]}


async def process_photo(row, client, http, tracker):
    import photo_local
    from photo_qc import split_urls
    paths = list(row.get("_local_photos") or [])[:C.MAX_PHOTOS_PER_ROW]
    for u in split_urls(row.get("media_urls"))[:max(0, C.MAX_PHOTOS_PER_ROW - len(paths))]:
        try:
            paths.append(await fetch(http, u, C.PHOTO_CACHE_DIR, C.PHOTO_TIMEOUT_SEC))
        except Unavailable:
            pass
    if not paths:
        return {"photo_status": "Unavailable", "photo_error": "no photos"}
    try:
        return await asyncio.to_thread(photo_local.analyse, paths, row)
    except Exception as e:
        return {"photo_status": "Error", "photo_error": f"{type(e).__name__}: {e}"[:200]}
