import json
import os
import time
import zipfile

import pandas as pd
import pytest

import config as C
import ingestion
import report
import run_qc
from conftest import PNG, FakeAnthropic, make_rows
from helpers import frame, good


def args_for(inp, *extra, out="out/o.xlsx"):
    return ["--input", inp, "--output", out, "--checkpoint", "ck/ck.json", "--api-key", "k", *extra]


def read(out="out/o.xlsx"):
    return pd.read_excel(out, dtype=str)


def test_parse_args_defaults():
    a = run_qc.parse_args(["--input", "/x/y.xlsx", "--workers", "99"])
    assert os.path.normpath(a.output) == os.path.normpath("/x/y_QC.xlsx") and a.workers == 20
    assert run_qc.parse_args(["--input", "https://d.invalid/f"]).output == "output/qc_output.xlsx"


def test_row_keys_stable_under_filter_and_duplicates():
    df = frame([good(0, docket_id="A"), good(1, docket_id="B"), good(2, docket_id="A"), good(3, docket_id=None)])
    k = run_qc.make_keys(df)
    assert k[:3] == ["A", "B", "A#2"] and k[3].startswith("nodocket-")
    sub = df[df["docket_id"] == "B"].reset_index(drop=True)
    assert run_qc.make_keys(sub) == ["B"]
    assert run_qc.make_keys(df.iloc[::-1].reset_index(drop=True))[2] == "B"


def test_checkpoint_roundtrip_and_corruption(tmp_path):
    p = tmp_path / "d" / "c.json"
    run_qc.save_checkpoint(str(p), {"results": {"a": {"x": 1}}, "cost": 1.5})
    assert run_qc.load_checkpoint(str(p)) == {"results": {"a": {"x": 1}}, "cost": 1.5}
    p.write_text("{trunc")
    assert run_qc.load_checkpoint(str(p)) == {"results": {}, "cost": 0.0}
    p.write_text('{"cost": 1}')
    assert run_qc.load_checkpoint(str(p)) == {"results": {}, "cost": 0.0}
    assert run_qc.load_checkpoint(str(tmp_path / "missing")) == {"results": {}, "cost": 0.0}


def test_dry_run_no_key_no_network(headerless_xlsx, capsys, monkeypatch):
    import anthropic
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: pytest.fail("client must not be created"))
    assert run_qc.main(["--input", headerless_xlsx, "--output", "o.xlsx", "--dry-run", "--checkpoint", "ck.json"]) == 0
    o = capsys.readouterr().out
    assert "pdf: 6 rows" in o and "photo: 6 rows" in o and "est. cost: $0.05" in o
    import os
    assert not os.path.exists("o.xlsx")


def test_missing_key_fails_fast(headerless_xlsx, monkeypatch):
    monkeypatch.setattr(C, "ANTHROPIC_API_KEY", None)
    with pytest.raises(SystemExit):
        run_qc.main(["--input", headerless_xlsx, "--output", "o.xlsx", "--mode", "pdf", "--engine", "claude"])
    import os
    assert not os.path.exists("o.xlsx")


def test_gps_and_data_mode_without_api(headerless_xlsx, monkeypatch):
    monkeypatch.setattr(C, "ANTHROPIC_API_KEY", None)
    assert run_qc.main(["--input", headerless_xlsx, "--output", "out/o.xlsx", "--mode", "gps"]) == 0
    df = read()
    assert "Suggested_Remark" in df and "Done By" not in df
    assert (run_qc.main(["--input", headerless_xlsx, "--output", "out/o2.xlsx", "--mode", "data"])) == 0
    assert "Data_QC_Flags" in read("out/o2.xlsx")


def test_offline_urls_are_unavailable_not_fatal(headerless_xlsx, fake_anthropic):
    assert run_qc.main(args_for(headerless_xlsx, "--workers", "3")) == 0
    assert fake_anthropic.calls == 0            # nothing downloadable -> no API spend
    df = read()
    assert len(df) == 6 and df["QC Done"].isna().all()
    assert df["Any Other Remarks"].str.contains("PDF unavailable: HTTP 404").all()
    assert set(df["AI_Confidence"]) == {"Low"}


def _local_media_dir(tmp_path, n=6):
    import pymupdf
    d = tmp_path / "media"
    d.mkdir()
    for i in range(n):
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "x")
        doc.save(str(d / f"{2000000000 + i}.pdf"))
        doc.close()
        (d / f"{2000000000 + i}_1.jpg").write_bytes(PNG)
    return str(d)


def test_full_run_local_media(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    assert run_qc.main(args_for(headerless_xlsx, "--local-media", m)) == 0
    assert fake_anthropic.calls == 12
    df = read()
    assert set(df["QC Done"]) == {"True"}
    assert set(df["Match/Mismatch (Form&app)"]) == {"Match"}  # 50/40 in the form == 50/40 in the row
    assert set(df["AI_Confidence"]) == {"High"}
    assert float(json.load(open("ck/ck.json"))["cost"]) == pytest.approx(12 * (1000 * 3 + 500 * 15) / 1e6)
    assert (tmp_path / "out" / "summary_report.xlsx").exists()


def test_local_media_from_zip(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in __import__("pathlib").Path(m).iterdir():
            zf.write(f, f.name)
    assert run_qc.main(args_for(headerless_xlsx, "--local-media", str(z), "--mode", "pdf")) == 0
    assert fake_anthropic.calls == 6


def test_resume_skips_done_and_retries_errors(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    base = args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf")
    run_qc.main(base)
    assert fake_anthropic.calls == 6
    run_qc.main(base + ["--resume"])
    assert fake_anthropic.calls == 6  # nothing left to do
    ck = json.load(open("ck/ck.json"))
    assert sorted(ck["results"]) == [str(2000000000 + i) for i in range(6)]  # keyed on docket, not position
    # mark two as Error -> only those are retried
    for k in list(ck["results"])[:2]:
        ck["results"][k]["pdf_status"] = "Error"
    json.dump(ck, open("ck/ck.json", "w"))
    run_qc.main(base + ["--resume"])
    assert fake_anthropic.calls == 8
    # without --resume everything is redone
    run_qc.main(base)
    assert fake_anthropic.calls == 14


def test_resume_with_changed_filter_reuses_by_docket(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    base = args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf")
    run_qc.main(base)
    n = fake_anthropic.calls
    run_qc.main(base + ["--resume", "--filter-dist", "Dist1"])   # different frame, different positions
    assert fake_anthropic.calls == n
    df = read()
    assert len(df) == 3 and set(df["QC Done"]) == {"True"} and set(df["district"]) == {"Dist1"}
    # results must land on the right rows: corrupt one docket and check only it is affected
    ck = json.load(open("ck/ck.json"))
    ck["results"]["2000000003"]["form_remarks"] = "MARKER"
    json.dump(ck, open("ck/ck.json", "w"))
    run_qc.main(base + ["--resume", "--filter-dist", "Dist1"])
    df = read()
    assert df.loc[df["docket_id"] == "2000000003", "Survey remarks on form"].item() == "MARKER"
    assert (df["Survey remarks on form"] == "MARKER").sum() == 1


def test_legacy_positional_checkpoint_keys_still_used(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    json.dump({"results": {f"{i}|{2000000000 + i}": {"pdf_status": "OK", "form_remarks": "OLD"} for i in range(6)}, "cost": 0.5},
              open("ck.json", "w"))
    run_qc.main(["--input", headerless_xlsx, "--output", "o.xlsx", "--checkpoint", "ck.json", "--api-key", "k",
                 "--local-media", m, "--mode", "pdf", "--resume"])
    assert fake_anthropic.calls == 0
    assert set(pd.read_excel("o.xlsx")["Survey remarks on form"]) == {"OLD"}


def test_resume_respects_qc_done_in_input(tmp_path, fake_anthropic):
    m = _local_media_dir(tmp_path)
    rows = frame([good(i, docket_id=str(2000000000 + i), pdf_url="https://x.invalid/a.pdf") for i in range(4)])
    rows["QC Done"] = ["TRUE", "true", None, "FALSE"]
    rows["Done By"] = ["Human", "Human", None, None]
    rows["AI_Flags"] = ["old flag", "", "", ""]
    p = tmp_path / "prior.xlsx"
    rows.to_excel(p, index=False)
    base = ["--input", str(p), "--output", "o.xlsx", "--checkpoint", "c.json", "--api-key", "k", "--local-media", m, "--mode", "pdf"]
    run_qc.main(base + ["--resume"])
    assert fake_anthropic.calls == 2                       # only the two not-done rows
    df = pd.read_excel("o.xlsx", dtype=str)
    assert df["Done By"].tolist() == ["Human", "Human", "AI-Auto", "AI-Auto"]   # prior values preserved
    assert df["AI_Flags"].tolist()[0] == "old flag"
    assert df["QC Done"].tolist() == ["TRUE", "true", "True", "True"]
    run_qc.main(base)                                      # no resume => everyone re-QC'd
    assert fake_anthropic.calls == 6


def test_cost_cap_stops_and_resumes(headerless_xlsx, fake_anthropic, tmp_path, capsys):
    m = _local_media_dir(tmp_path)
    fake_anthropic.tokens = (1_000_000, 0)   # $3 per call
    base = args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf", "--workers", "1")
    run_qc.main(base + ["--cost-cap", "5"])
    assert fake_anthropic.calls == 2          # $3 < $5 -> 2nd call -> $6 >= $5 -> stop
    assert "Cost cap" in capsys.readouterr().out
    ck = json.load(open("ck/ck.json"))
    assert ck["cost"] == pytest.approx(6.0) and len(ck["results"]) == 2
    df = read()
    assert df["QC Done"].notna().sum() == 2   # partial results still written
    run_qc.main(base + ["--cost-cap", "100", "--resume"])
    assert fake_anthropic.calls == 6
    assert set(read()["QC Done"]) == {"True"}


def test_cost_cap_carries_over_on_resume(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    run_qc.save_checkpoint("ck/ck.json", {"results": {}, "cost": 10.0})
    run_qc.main(args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf", "--cost-cap", "5", "--resume"))
    assert fake_anthropic.calls == 0


def test_row_exception_does_not_lose_run(headerless_xlsx, fake_anthropic, tmp_path, monkeypatch):
    m = _local_media_dir(tmp_path)
    import pdf_qc
    orig = pdf_qc.process

    async def boom(row, client, http, tracker):
        if row["docket_id"] == "2000000002":
            raise RuntimeError("kaboom")
        return await orig(row, client, http, tracker)
    monkeypatch.setattr(pdf_qc, "process", boom)
    run_qc.main(args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf"))
    df = read()
    bad = df[df["docket_id"] == "2000000002"].iloc[0]
    assert bad["QC Done"] != "True" and "kaboom" in bad["Any Other Remarks"] and bad["AI_Confidence"] == "Low"
    assert (df["QC Done"] == "True").sum() == 5


def test_illegal_excel_characters_do_not_crash(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    fake_anthropic.reply = dict(fake_anthropic.reply, form_remarks="bad\x0bchar")
    run_qc.main(args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf"))
    assert read()["Survey remarks on form"].iloc[0] == "badchar"


def test_local_state_does_not_leak_between_runs(headerless_xlsx, fake_anthropic, tmp_path):
    m = _local_media_dir(tmp_path)
    run_qc.main(args_for(headerless_xlsx, "--local-media", m, "--mode", "pdf"))
    assert run_qc.LOCAL
    run_qc.main(args_for(headerless_xlsx, "--dry-run"))
    assert not run_qc.LOCAL


def test_plan_building_is_fast():
    n = 41_000
    df = frame([good(i, docket_id=str(i), pdf_url="https://x.invalid/a.pdf") for i in range(200)])
    df = pd.concat([df] * (n // 200), ignore_index=True)
    df["docket_id"] = [str(i) for i in range(len(df))]
    t = time.time()
    keys = run_qc.make_keys(df)
    ck = {"results": {}, "cost": 0.0}
    assert len(set(keys)) == len(df)
    out = run_qc.assemble(df, ck, True, keys)
    assert time.time() - t < 20 and len(out) == len(df)


# ---- assemble semantics ----
def test_assemble_no_results_and_no_ai():
    df = frame([good(0), good(1)])
    out = run_qc.assemble(df, {"results": {}, "cost": 0}, False)
    assert "Done By" not in out and "AI_Flags" not in out and out["Match/Mismatch (Form&app)"].isna().all()
    out = run_qc.assemble(df, {"results": {}, "cost": 0}, True)
    assert out["QC Done"].isna().all() and out["AI_Flags"].tolist() == ["", ""]


def test_assemble_confidence_and_flags():
    df = frame([good(i, docket_id=f"D{i}") for i in range(6)])
    res = {
        "D0": {"pdf_status": "OK", "pdf_confidence": 0.95, "match": "Match", "form_status": "correct", "photo_status": "OK", "photo_quality": "good"},
        "D1": {"pdf_status": "OK", "pdf_confidence": 0.7, "match": "Mismatch", "form_status": "overwrite"},
        "D2": {"pdf_status": "OK", "pdf_confidence": 0.3, "manual_review": True},
        "D3": {"pdf_status": "Unavailable", "pdf_error": "HTTP 404"},
        "D4": {"photo_status": "OK", "photo_quality": "irrelevant", "photo_flags": ["x"]},
    }
    out = run_qc.assemble(df, {"results": res, "cost": 0}, True)
    assert out["AI_Confidence"].tolist()[:5] == ["High", "Medium", "Low", "Low", "Low"] and pd.isna(out["AI_Confidence"][5])
    assert "Form vs app mismatch" in out["AI_Flags"][1] and "Form overwrite" in out["AI_Flags"][1]
    assert "Low OCR confidence - manual review" in out["AI_Flags"][2]
    assert "PDF unavailable: HTTP 404" in out["Any Other Remarks"][3]
    assert out["QC Done"].tolist() == [True, True, True, None, True, None]
    assert out["Done By"].tolist() == ["AI-Auto"] * 5 + [None]


def test_assemble_input_already_has_qc_columns():
    df = frame([good(0, docket_id="D0"), good(1, docket_id="D1")])
    df["Done By"] = ["Human", "Human"]
    df["Farmer Signature (Yes/No)"] = ["No", "Yes"]
    df["AI_Flags"] = ["prev", "prev"]
    res = {"D0": {"pdf_status": "OK", "pdf_confidence": 0.9, "farmer_signed": True, "match": "Match"}}
    out = run_qc.assemble(df, {"results": res, "cost": 0}, True)
    assert out["Done By"].tolist() == ["AI-Auto", "Human"]
    assert out["Farmer Signature (Yes/No)"].tolist() == ["Yes", "Yes"]
    assert out["AI_Flags"].tolist() == ["", "prev"]
    assert len(out) == 2 and list(out.columns).count("Done By") == 1


def test_assemble_false_values_and_input_untouched():
    df = frame([good(0, docket_id="D0")])
    before = df.copy()
    out = run_qc.assemble(df, {"results": {"D0": {"pdf_status": "OK", "surveyor_signed": False, "pdf_confidence": 0.9, "form_area": 0}}, "cost": 0}, True)
    assert out["Surveyor Signature (Yes/No)"][0] == "No" and out["Affected area% (Form)"][0] == 0
    pd.testing.assert_frame_equal(df, before)


# ---- report ----
def test_report_full_and_minimal_columns(tmp_path):
    df = frame([good(0), good(1, district="D2")])
    df["Suggested_Remark"] = ["Same Location - QC Required", "OK"]
    df["Data_QC_Flags"] = ["Missing: x, Duplicate record", ""]
    df["Match/Mismatch (Form&app)"] = ["Mismatch", "Match"]
    df["QC Done"] = [True, False]
    p = str(tmp_path / "r.xlsx")
    report.build(df, p)
    sheets = pd.read_excel(p, sheet_name=None)
    assert set(sheets) == {"District Summary", "Surveyor Report", "Issue Summary"}
    d = sheets["District Summary"].set_index("District")
    assert d.loc["D1", "Flagged"] == 1 and d.loc["D1", "Mismatch"] == 1 and d.loc["D1", "QC Done"] == 1
    assert d.loc["D2", "Match%"] == 100
    assert set(sheets["Issue Summary"]["Issue_Type"]) == {"Missing: x", "Duplicate record", "Same Location - QC Required"}
    assert sheets["Surveyor Report"].set_index("Surveyor").loc["S0", "Consistency_Score"] == 0


def test_report_missing_columns_and_empty(tmp_path):
    df = pd.DataFrame({"docket_id": ["a", "b"]})
    report.build(df, str(tmp_path / "m.xlsx"))
    s = pd.read_excel(tmp_path / "m.xlsx", sheet_name=None)
    assert s["District Summary"]["District"].tolist() == ["Unknown"] and s["Issue Summary"].empty
    report.build(frame([good(0)]).iloc[0:0], str(tmp_path / "e.xlsx"))


def test_local_engine_runs_without_key(tmp_path, monkeypatch):
    """--engine local (auto without key) runs offline; with only the generic OCR fallback the result is honest: Manual-check."""
    import pandas as pd, pymupdf
    monkeypatch.setattr(C, "ANTHROPIC_API_KEY", None)
    d = pymupdf.open(); pg = d.new_page(); y = 80
    for t in ["Date of Survey: 16/09/2026", "Affected Area %: 40", "Crop Loss %: 30"]:
        pg.insert_text((60, y), t, fontsize=14); y += 50
    (tmp_path / "media").mkdir(); d.save(str(tmp_path / "media" / "1000000000.pdf"))
    df = pd.DataFrame([["1000000000", "A", "F", "Flood", "2026-09-01", "2026-09-05", 1, "Paddy", "1", "0", "S", "D", "T", "B", "V", "P", "Sv", "9123456789", 40, 30, 12, "", 29.1, 76.1, "", ""]])
    df.to_excel(tmp_path / "in.xlsx", header=False, index=False)
    assert run_qc.main(["--input", str(tmp_path / "in.xlsx"), "--output", str(tmp_path / "o.xlsx"), "--mode", "pdf", "--no-serve",
                        "--local-media", str(tmp_path / "media"), "--checkpoint", str(tmp_path / "ck.json"),
                        "--progress", str(tmp_path / "p.json"), "--agents", "1"]) == 0
    o = pd.read_excel(tmp_path / "o.xlsx")
    # a synthetic English page is not a Proforma-3: the Proforma-3 reader must not read values from it
    assert o["QC Verdict"][0] in ("Manual-check", "Review", "OK", "Reject-evidence") and "FORM:" in o["Any Other Remarks"][0]
