import asyncio
import json
import zipfile
from types import SimpleNamespace

import httpx
import pymupdf
import pytest

import common
import local_media
import pdf_qc
import photo_qc
from conftest import PNG, FakeAnthropic
from helpers import frame, good

G1 = "06DAD6A0-AA87-4006-9717-B4CAA9805548"
G2 = "11111111-2222-3333-4444-555555555555"


def make_pdf(path):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "form")
    doc.save(str(path))
    doc.close()


def media_df():
    return frame([good(0, docket_id="1000000001"),
                  good(1, docket_id="1000000002", pdf_url=f"https://x.invalid/dl?mediaID={G1}.pdf",
                       media_urls=f"https://x.invalid/dl?mediaID={G2}_x000D_\nhttps://x.invalid/other"),
                  good(2, docket_id=None)])


def test_split_urls_x000d_junk():
    v = "https://a.invalid/1.jpg_x000D_\nhttps://a.invalid/2.jpg;https://a.invalid/3.jpg, junk"
    assert photo_qc.split_urls(v) == ["https://a.invalid/1.jpg", "https://a.invalid/2.jpg", "https://a.invalid/3.jpg"]
    assert photo_qc.split_urls(None) == [] and photo_qc.split_urls("_x000D_") == []
    assert photo_qc.split_urls(float("nan")) == []  # 'nan' string has no http


def _tree(root):
    root.mkdir(parents=True, exist_ok=True)
    make_pdf(root / "1000000001.pdf")
    (root / "1000000001_1.jpg").write_bytes(PNG)
    (root / "sub").mkdir()
    (root / "sub" / f"{G1}.pdf").write_bytes(b"%PDF")
    (root / "sub" / f"{G2.lower()}.jpg").write_bytes(PNG)
    (root / "1000000002").mkdir()
    (root / "1000000002" / "photo2.png").write_bytes(PNG)
    (root / "unrelated.pdf").write_bytes(b"x")
    (root / "notes.txt").write_text("x")
    (root / "10000000011.jpg").write_bytes(PNG)  # longer number must not match docket ...001


def _check(idx):
    assert set(idx) == {"1000000001", "1000000002"}
    assert [p.name for p in idx["1000000001"]["pdfs"]] == ["1000000001.pdf"]
    assert [p.name for p in idx["1000000001"]["images"]] == ["1000000001_1.jpg"]
    assert {p.name for p in idx["1000000002"]["pdfs"]} == {f"{G1}.pdf"}
    assert {p.name for p in idx["1000000002"]["images"]} == {f"{G2.lower()}.jpg", "photo2.png"}


def test_local_media_folder(tmp_path):
    _tree(tmp_path / "media")
    _check(local_media.index(str(tmp_path / "media"), media_df()))


def test_local_media_zip_and_nested_zip(tmp_path):
    _tree(tmp_path / "src")
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for f in (tmp_path / "src").rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(tmp_path / "src"))
    _check(local_media.index(str(z), media_df()))
    # nested zip + corrupt nested zip
    outer = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.write(z, "inner.zip")
        zf.writestr("bad.zip", b"not a zip")
    _check(local_media.index(str(outer), media_df()))


def test_local_media_same_named_zips_do_not_collide(tmp_path):
    for name, docket in (("a", "1000000001"), ("b", "1000000002")):
        d = tmp_path / name
        d.mkdir()
        with zipfile.ZipFile(d / "media.zip", "w") as zf:
            zf.writestr(f"{docket}.pdf", b"%PDF")
    assert set(local_media.index(str(tmp_path / "a" / "media.zip"), media_df())) == {"1000000001"}
    assert set(local_media.index(str(tmp_path / "b" / "media.zip"), media_df())) == {"1000000002"}


def test_local_media_bad_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        local_media.index(str(tmp_path / "nope.zip"), media_df())


# ---- common ----
def test_parse_json_variants():
    assert common.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert common.parse_json('Sure! here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(json.JSONDecodeError):
        common.parse_json("no json here")


def test_image_media_type():
    assert common.image_media_type(PNG) == "image/png"
    assert common.image_media_type(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert common.image_media_type(b"RIFFxxxxWEBPzz") == "image/webp"
    assert common.image_media_type(b"GIF89a") == ""


def test_cost_tracker():
    t = common.CostTracker(cap=0.01)
    t.add("m", SimpleNamespace(input_tokens=1_000_000, output_tokens=0))
    assert t.spent == pytest.approx(3.0) and t.exceeded
    assert not common.CostTracker().exceeded


def client_returning(*texts, exc=None):
    seq = list(texts)

    async def create(**kw):
        client.n += 1
        item = seq.pop(0) if len(seq) > 1 else seq[0]
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=item)],
                               usage=SimpleNamespace(input_tokens=10, output_tokens=10))
    client = SimpleNamespace(messages=SimpleNamespace(create=create), n=0)
    return client


def test_call_claude_retries_and_non_object():
    c = client_returning(RuntimeError("overloaded"), "garbage", '{"ok": true}')
    assert asyncio.run(common.call_claude(c, common.CostTracker(), "m", [])) == {"ok": True}
    assert c.n == 3
    c = client_returning("[1, 2]")
    with pytest.raises(json.JSONDecodeError):
        asyncio.run(common.call_claude(c, common.CostTracker(), "m", [], retries=2))
    assert c.n == 2


def test_call_claude_no_retry_on_auth_error():
    class E(Exception):
        status_code = 401
    c = client_returning(E("bad key"))
    with pytest.raises(E):
        asyncio.run(common.call_claude(c, common.CostTracker(), "m", []))
    assert c.n == 1


# ---- pdf_qc / photo_qc ----
def run(coro):
    return asyncio.run(coro)


async def _http_ctx(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def process(mod, row, client=None, handler=None):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler or (lambda r: httpx.Response(404)))) as http:
            return await mod.process(row, client or FakeAnthropic(), http, common.CostTracker())
    return run(go())


def test_pdf_no_url_and_http_errors(tmp_path):
    assert process(pdf_qc, {"pdf_url": None})["pdf_status"] == "Unavailable"
    assert process(pdf_qc, {"pdf_url": float("nan")})["pdf_error"] == "no url"
    r = process(pdf_qc, {"pdf_url": "https://x.invalid/a.pdf"})
    assert r == {"pdf_status": "Unavailable", "pdf_error": "HTTP 404"}
    r = process(pdf_qc, {"pdf_url": "https://x.invalid/a.pdf"}, handler=lambda q: (_ for _ in ()).throw(httpx.ConnectError("x")))
    assert r["pdf_status"] == "Unavailable"


def test_pdf_403_retries_with_referer():
    seen = []

    def h(req):
        seen.append(req.headers.get("referer"))
        return httpx.Response(200 if req.headers.get("referer") else 403, content=b"x")
    r = process(pdf_qc, {"pdf_url": "https://x.invalid/a.pdf_x000D_\n"}, handler=h)
    assert seen == [None, "https://pmfby.gov.in"]
    assert r["pdf_error"].startswith("render")  # content is not a real pdf


def test_pdf_render_failure_and_success_and_match(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not pdf")
    assert process(pdf_qc, {"_local_pdfs": [bad]})["pdf_error"].startswith("render")
    good_pdf = tmp_path / "ok.pdf"
    make_pdf(good_pdf)
    FakeAnthropic.reply = dict(FakeAnthropic.reply)
    r = process(pdf_qc, {"_local_pdfs": [good_pdf], "affected_area_pct": 52, "crop_loss_pct": 38})
    assert r["pdf_status"] == "OK" and r["match"] == "Match" and not r["manual_review"] and r["farmer_signed"] is True
    assert process(pdf_qc, {"_local_pdfs": [good_pdf], "affected_area_pct": 90, "crop_loss_pct": 38})["match"] == "Mismatch"
    assert process(pdf_qc, {"_local_pdfs": [good_pdf]})["match"] == "NA"


def test_pdf_api_error_and_bad_status_and_low_conf(tmp_path):
    p = tmp_path / "ok.pdf"
    make_pdf(p)

    class E(Exception):
        status_code = 400
    r = process(pdf_qc, {"_local_pdfs": [p]}, client=client_returning(E("boom")))
    assert r["pdf_status"] == "Error" and "boom" in r["pdf_error"]
    c = client_returning(json.dumps({"confidence": 7, "form_status": "weird", "affected_area_percent": "x"}))
    r = process(pdf_qc, {"_local_pdfs": [p]}, client=c)
    assert r["form_status"] == "incomplete" and r["pdf_confidence"] == 1.0 and r["match"] == "NA"
    c = client_returning(json.dumps({"confidence": 0.3}))
    assert process(pdf_qc, {"_local_pdfs": [p]}, client=c)["manual_review"] is True


def test_photo_paths(tmp_path):
    assert process(photo_qc, {"media_urls": None})["photo_error"] == "no urls"
    r = process(photo_qc, {"media_urls": "https://x.invalid/a.jpg"})
    assert r["photo_error"] == "no downloadable photos"
    # non-image bytes and oversize are dropped
    r = process(photo_qc, {"media_urls": "https://x.invalid/a.jpg"}, handler=lambda q: httpx.Response(200, content=b"<html>"))
    assert r["photo_status"] == "Unavailable"
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    r = process(photo_qc, {"_local_photos": [img], "crop_loss_pct": 40, "latitude": 29.0, "longitude": 76.0,
                           "survey_start_date": "2026-09-01"})
    assert r["photo_status"] == "OK" and r["field_photo"] == "standing crop" and r["photo_flags"] == []


def test_photo_flags_and_malformed_reply(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(PNG)
    reply = {"field_photo_type": "bogus", "estimated_loss_pct": 90, "photo_date": "15122025",
             "photo_gps": {"lat": 30.0, "lng": 76.0}, "flags": "single string"}
    row = {"_local_photos": [img], "crop_loss_pct": 40, "latitude": 29.0, "longitude": 76.0, "survey_start_date": "2026-09-01",
           "survey_end_date": "2026-09-05"}
    r = process(photo_qc, row, client=client_returning(json.dumps(reply)))
    assert r["field_photo"] is None
    assert {"single string", "GPS mismatch > 200m", "Photo date outside survey period", "Loss estimate differs from app"} <= set(r["photo_flags"])
    reply.update(photo_gps="oops", flags=None)
    r = process(photo_qc, row, client=client_returning(json.dumps(reply)))
    assert r["photo_status"] == "OK"
    r = process(photo_qc, row, client=client_returning("[1]"))  # non-object reply must not crash
    assert r["photo_status"] == "Error"


def test_photo_invalid_url_does_not_raise():
    r = process(photo_qc, {"media_urls": "http://[bad"})
    assert r["photo_status"] == "Unavailable"


def test_photo_max_photos(tmp_path):
    import config as C
    imgs = []
    for i in range(8):
        p = tmp_path / f"{i}.png"
        p.write_bytes(PNG)
        imgs.append(p)
    captured = {}

    async def create(**kw):
        captured["n"] = sum(b["type"] == "image" for b in kw["messages"][0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="{}")], usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    process(photo_qc, {"_local_photos": imgs}, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    assert captured["n"] == C.MAX_PHOTOS_PER_ROW
