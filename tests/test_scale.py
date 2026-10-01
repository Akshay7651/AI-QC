"""Scale features: sqlite checkpoint, --chunk-rows, --offset/--limit, --discard-media, kill -9 + resume, dry-run estimate."""
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

import ckstore
import local_engine
import run_qc
from conftest import PNG, ROOT, make_rows
from test_orch import D, argv, media, xlsx


@pytest.fixture(autouse=True)
def _fake_env(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MARK_DIR", str(tmp_path))
    monkeypatch.delenv("FAKE_SLEEP", raising=False)


def n_rows_file(tmp_path, n, name="big.xlsx"):
    p = tmp_path / name
    pd.DataFrame(make_rows(n)).to_excel(p, header=False, index=False)
    return str(p)


def dockets(n):
    return [str(2000000000 + i) for i in range(n)]


# ------------------------------------------------------------------------------------------------ store
def test_store_roundtrip_done_keys_and_legacy_import(tmp_path):
    st = ckstore.ResultStore(tmp_path / "s.sqlite", fresh=True)
    st["a"] = {"verdict": "OK", "remark": "Hindi किसान"}
    st["b"] = {"form": {"x": 1}}                       # not finished (no verdict)
    assert st.done_keys() == {"a"} and "b" in st and len(st) == 2
    assert st.get("a")["remark"].startswith("Hindi") and st.get_many(["b", "zz", "a"])[1] is None
    st.close()
    st2 = ckstore.ResultStore(tmp_path / "s.sqlite")    # reopened: same content, done set rebuilt
    assert st2.done_keys() == {"a"}
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps({"results": {"c": {"verdict": "Review"}, "a": {"verdict": "CHANGED"}}, "cost": 2.5}))
    assert st2.import_legacy_json(legacy) == 1          # 'a' is newer in the store and is kept
    assert st2.get("a")["verdict"] == "OK" and st2.get("c")["verdict"] == "Review" and st2.meta_get("cost") == "2.5"
    assert st2.import_legacy_json(legacy) == 0          # idempotent


def test_old_json_checkpoint_is_migrated_on_resume(tmp_path):
    import fakes_engine
    inp, m = xlsx(tmp_path), media(tmp_path)
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk")) == 0
    st = ckstore.ResultStore("ck/ck.sqlite")
    old = {"results": {k: v for k, v in st.items()}, "cost": 0.0}
    st.close()
    for suf in ("", "-wal", "-shm"):
        if os.path.exists("ck/ck.sqlite" + suf):
            os.remove("ck/ck.sqlite" + suf)
    json.dump(old, open("ck/ck.json", "w"))
    fakes_engine.CALLS.update(form=0, photo=0)
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--resume")) == 0
    assert fakes_engine.CALLS == {"form": 0, "photo": 0}
    assert pd.read_excel("out/o.xlsx", dtype=str)["QC Verdict"].notna().all()


# ------------------------------------------------------------------------------------------------ chunking
def test_chunked_output_parts_and_merged_csv(tmp_path):
    n = 7
    inp = n_rows_file(tmp_path, n)
    m = media(tmp_path, dockets(n))
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--agents", "2", "--no-risk", "--chunk-rows", "3")) == 0
    parts = sorted(Path("out").glob("o_part*.xlsx"))
    assert [p.name for p in parts] == ["o_part001.xlsx", "o_part002.xlsx", "o_part003.xlsx"]
    got = pd.concat([pd.read_excel(p, dtype=str) for p in parts], ignore_index=True)
    assert got["docket_id"].tolist() == dockets(n) and got["QC Verdict"].notna().all()
    assert not Path("out/o.xlsx").exists()                       # chunk mode writes parts + csv, not one file
    raw = Path("out/o.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                         # BOM so Excel opens Hindi text
    c = pd.read_csv("out/o.csv", dtype=str, encoding="utf-8-sig")
    assert len(c) == n and c["docket_id"].tolist() == dockets(n) and list(c.columns) == list(got.columns)
    assert c["QC Verdict"].tolist() == got["QC Verdict"].tolist()
    assert Path("out/summary_report.xlsx").exists()
    assert json.load(open("out/progress.json"))["status"] == "Done"


def test_chunked_never_rewrites_finished_parts_and_resume_is_a_noop(tmp_path, monkeypatch):
    import fakes_engine
    n = 6
    inp = n_rows_file(tmp_path, n)
    m = media(tmp_path, dockets(n))
    monkeypatch.setenv("FAKE_SLEEP", "0.25")
    res = {}
    th = threading.Thread(target=lambda: res.update(rc=run_qc.main(
        argv(inp, "--local-media", m, "--inline", "--agents", "1", "--autosave-sec", "0.4", "--no-risk", "--chunk-rows", "2"))))
    th.start()
    p1 = Path("out/o_part001.xlsx")
    seen_m = None
    t_end = time.time() + 60
    while th.is_alive() and time.time() < t_end:
        st_path = Path("ck/ck.sqlite")
        if p1.exists() and st_path.exists():
            st = ckstore.ResultStore(st_path)
            fin = st.meta_get("part:out/o:0")
            st.close()
            if fin == "done":
                seen_m = p1.stat().st_mtime_ns
                break
        time.sleep(0.05)
    th.join(120)
    assert res["rc"] == 0 and seen_m is not None, "part 1 never reported finished while the run was going"
    assert p1.stat().st_mtime_ns == seen_m                         # later autosaves/final save did not touch the finished part
    mt = {p.name: p.stat().st_mtime_ns for p in Path("out").glob("o_part*.xlsx")}
    fakes_engine.CALLS.update(form=0, photo=0)
    monkeypatch.delenv("FAKE_SLEEP")
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--chunk-rows", "2", "--resume")) == 0
    assert fakes_engine.CALLS == {"form": 0, "photo": 0}
    assert {p.name: p.stat().st_mtime_ns for p in Path("out").glob("o_part*.xlsx")} == mt   # resume rewrote nothing


def test_chunked_partial_run_then_resume_completes(tmp_path, monkeypatch):
    n = 6
    inp = n_rows_file(tmp_path, n)
    m = media(tmp_path, dockets(n))
    calls = {"n": 0}
    real = local_engine.LocalRunner._finalize

    def fin(self, i):
        real(self, i)
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
    monkeypatch.setattr(local_engine.LocalRunner, "_finalize", fin)
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--agents", "1", "--no-risk", "--chunk-rows", "2")) == 130
    monkeypatch.setattr(local_engine.LocalRunner, "_finalize", real)
    c = pd.read_csv("out/o.csv", dtype=str, encoding="utf-8-sig")
    assert 1 <= c["QC Verdict"].notna().sum() < n                  # partial CSV exists after Ctrl-C
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--chunk-rows", "2", "--resume")) == 0
    c = pd.read_csv("out/o.csv", dtype=str, encoding="utf-8-sig")
    assert len(c) == n and c["QC Verdict"].notna().all()


# ------------------------------------------------------------------------------------------------ offset / limit
def test_offset_limit_batch_with_resume(tmp_path):
    import fakes_engine
    n = 8
    inp = n_rows_file(tmp_path, n)
    m = media(tmp_path, dockets(n))
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--offset", "2", "--limit", "3")) == 0
    df = pd.read_excel("out/o.xlsx", dtype=str)
    assert df["docket_id"].tolist() == dockets(n)[2:5] and df["QC Verdict"].notna().all()
    fakes_engine.CALLS.update(form=0, photo=0)
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--offset", "2", "--limit", "3", "--resume")) == 0
    assert fakes_engine.CALLS == {"form": 0, "photo": 0}
    # next batch, other output: only its own rows are processed
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--offset", "5", "--limit", "10", "--resume", out="out/b2.xlsx")) == 0
    assert pd.read_excel("out/b2.xlsx", dtype=str)["docket_id"].tolist() == dockets(n)[5:]
    assert fakes_engine.CALLS["form"] == 3


def test_offset_batches_see_whole_file_for_gps(tmp_path):
    """Same-location counts must not depend on how the file is cut into batches."""
    rows = make_rows(6)
    for r in rows:
        r[22], r[23], r[16], r[17] = 29.0, 76.0, "Same Surveyor", "9000000000"   # one surveyor, one spot
    p = tmp_path / "same.xlsx"
    pd.DataFrame(rows).to_excel(p, header=False, index=False)
    m = media(tmp_path, dockets(6))
    assert run_qc.main(argv(str(p), "--local-media", m, "--inline", "--no-risk", "--offset", "4", "--limit", "2")) == 0
    df = pd.read_excel("out/o.xlsx")
    assert len(df) == 2
    assert (df["Nearby_Same_Surveyor_25m"].astype(float) == 5).all()      # 5 neighbours: counted over the whole file, not the 2-row batch


# ------------------------------------------------------------------------------------------------ discard-media
def _serve_png(monkeypatch):
    prev = httpx.AsyncClient.__init__
    hits = []

    def handler(req):
        hits.append(str(req.url))
        return httpx.Response(200, content=PNG)

    def init(self, *a, **k):
        k["transport"] = httpx.MockTransport(handler)
        prev(self, *a, **k)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    return hits


def _cache_files():
    return [p for d in ("cache/pdfs", "cache/photos") for p in Path(d).glob("*")] if Path("cache").exists() else []


def test_discard_media_deletes_downloads_but_keeps_cache_without_flag(tmp_path, monkeypatch):
    hits = _serve_png(monkeypatch)
    inp = xlsx(tmp_path)
    assert run_qc.main(argv(inp, "--inline", "--agents", "2", "--no-risk")) == 0
    assert hits and len(_cache_files()) >= 6                       # without the flag the cache keeps growing
    import shutil
    shutil.rmtree("cache")
    hits.clear()
    assert run_qc.main(argv(inp, "--inline", "--agents", "2", "--no-risk", "--discard-media")) == 0
    assert hits and _cache_files() == []                           # every row's downloads are gone
    df = pd.read_excel("out/o.xlsx", dtype=str)
    assert df["QC Verdict"].notna().all() and not (df["Any Other Remarks"].str.contains("could not be retrieved", na=False)).any()


def test_discard_media_never_touches_local_media_or_input(tmp_path, monkeypatch):
    _serve_png(monkeypatch)
    inp = xlsx(tmp_path)
    m = media(tmp_path)
    before = sorted(os.listdir(m))
    assert run_qc.main(argv(inp, "--local-media", m, "--inline", "--no-risk", "--discard-media")) == 0
    assert sorted(os.listdir(m)) == before and Path(inp).exists()
    assert all((Path(m) / f).stat().st_size == len(PNG) for f in before)


def test_discard_media_shared_url_is_kept_until_last_row_done(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()
    f = cache / "abc"
    f.write_bytes(PNG)
    r = local_engine.LocalRunner([{}], ["k"], [0], {"results": {}}, ["form"], {}, {}, inline=True,
                                 cache_dirs={"form": str(cache), "photo": str(cache)}, discard_media=True)
    s1, s2 = {}, {}
    r._track(s1, f)
    r._track(s2, f)
    r._release(s1)
    assert f.exists()
    r._release(s2)
    assert not f.exists()
    other = tmp_path / "elsewhere.jpg"          # a file outside the cache dirs is never deleted, even if tracked by mistake
    other.write_bytes(PNG)
    s3 = {}
    r._track(s3, other)
    r._release(s3)
    assert other.exists()


# ------------------------------------------------------------------------------------------------ kill -9 and resume
def _spawn(inp, out, extra, env):
    cmd = [sys.executable, str(ROOT / "run_qc.py"), "--input", inp, "--output", out, "--checkpoint", "ck/ck.json",
           "--progress", "out/progress.json", "--no-serve", "--engine", "local", "--engine-module", "fakes_engine", "--rate", "0",
           "--no-risk", "--inline", "--agents", "2", *extra]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


@pytest.mark.skipif(sys.platform == "win32", reason="uses SIGKILL")
@pytest.mark.parametrize("chunk", [[], ["--chunk-rows", "4"]])
def test_kill_9_midway_then_resume_never_redoes_finished_rows(tmp_path, chunk):
    n = 14
    inp = n_rows_file(tmp_path, n)
    m = media(tmp_path, dockets(n))
    log = tmp_path / "calls.log"
    env = dict(os.environ, FAKE_SLEEP="0.4", FAKE_LOG=str(log), FAKE_MARK_DIR=str(tmp_path),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "tests")]))
    p = _spawn(inp, "out/o.xlsx", ["--local-media", m, "--autosave-sec", "1", *chunk], env)
    t_end = time.time() + 90
    done_at_kill = set()
    while time.time() < t_end and p.poll() is None:
        sp = Path("ck/ck.sqlite")
        if sp.exists():
            try:
                st = ckstore.ResultStore(sp)
                got = st.done_keys()
                st.close()
            except Exception:
                got = set()
            if len(got) >= 5:
                break
        time.sleep(0.1)
    os.killpg(p.pid, signal.SIGKILL)          # power loss: no cleanup code runs at all
    p.wait()
    st = ckstore.ResultStore("ck/ck.sqlite")  # the committed rows survive and the db opens cleanly
    done_at_kill = st.done_keys()
    st.close()
    assert 5 <= len(done_at_kill) < n
    first_run_calls = log.read_text().split()
    log.unlink()
    env2 = dict(env, FAKE_SLEEP="0")
    p2 = _spawn(inp, "out/o.xlsx", ["--local-media", m, "--resume", *chunk], env2)
    assert p2.wait(timeout=120) == 0
    second = log.read_text().split()
    assert not (set(second) & done_at_kill), "rows that were already finished were processed again"
    assert set(second) | done_at_kill == set(dockets(n)) and len(second) == len(set(second))
    if chunk:
        c = pd.read_csv("out/o.csv", dtype=str, encoding="utf-8-sig")
        assert c["docket_id"].tolist() == dockets(n) and c["QC Verdict"].notna().all()
    else:
        df = pd.read_excel("out/o.xlsx", dtype=str)
        assert len(df) == n and df["QC Verdict"].notna().all()


# ------------------------------------------------------------------------------------------------ dry-run estimate
def test_estimate_uses_agents_cores_rate_and_discard(tmp_path):
    n = 40
    p = n_rows_file(tmp_path, n)
    a = run_qc.parse_args(["--input", p, "--agents", "2", "--rate", "5", "--chunk-rows", "10", "--discard-media"])
    import ingestion
    df = ingestion.load(p)
    plan = {"pdf": list(range(n)), "photo": list(range(n))}
    e = run_qc.estimate(a, df, plan)
    cpu = os.cpu_count() or 4
    assert e["rows"] == n and e["eff"] == min(2, cpu) and e["chunks"] == 4
    assert e["t_cpu"] == pytest.approx(n * run_qc.CORE_SEC_PER_ROW / e["eff"])
    assert e["net_files"] == 2 * n and e["t_net"] == pytest.approx(2 * n / 5)      # 1 form + 1 photo url per row
    assert e["t"] == max(e["t_cpu"], e["t_net"])
    assert e["media_mb"] < 100                                                      # --discard-media: cache stays bounded
    a2 = run_qc.parse_args(["--input", p, "--agents", "2", "--local-media", "x"])
    e2 = run_qc.estimate(a2, df, plan)
    assert e2["net_files"] == 0 and e2["media_mb"] == 0


def test_dry_run_prints_time_and_disk(tmp_path, capsys):
    p = n_rows_file(tmp_path, 12)
    assert run_qc.main(["--input", p, "--output", "o.xlsx", "--dry-run", "--agents", "4", "--chunk-rows", "5", "--engine", "local"]) == 0
    o = capsys.readouterr().out
    assert "Offline engine estimate for 12 rows" in o and "total" in o and "disk:" in o and "3 parts" in o
    assert not Path("o.xlsx").exists() and not Path("output/checkpoint.sqlite").exists()
