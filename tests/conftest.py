import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Run every test in its own cwd (caches/checkpoints are relative) and never really sleep."""
    monkeypatch.chdir(tmp_path)
    real_sleep = asyncio.sleep

    async def fast(_s, *a, **k):
        await real_sleep(0)
    monkeypatch.setattr("common.asyncio.sleep", fast)
    import httpx
    orig_init = httpx.AsyncClient.__init__

    def no_network(self, *a, **k):  # every AsyncClient without an explicit transport answers 404 offline
        k.setdefault("transport", httpx.MockTransport(lambda req: httpx.Response(404)))
        orig_init(self, *a, **k)
    monkeypatch.setattr(httpx.AsyncClient, "__init__", no_network)
    import run_qc
    run_qc.LOCAL.clear()
    yield


def make_rows(n=6, **over):
    """Synthetic headerless CLAP rows (26 columns in DEFAULT_ORDER). No real data."""
    rows = []
    for i in range(n):
        r = [f"{2000000000 + i}", f"APP{i}", f"Farmer {i}", "Flood", "2026-09-01", "2026-09-05", 1.2, "Paddy",
             str(i + 1), "0", "StateA", f"Dist{i % 2}", "T1", "B1", f"V{i}", "P1", f"Surveyor {i}",
             f"9{123456789 + i}", 50, 40, 20, "", 29.0 + i * 0.01, 76.0 + i * 0.01,
             f"https://example.invalid/pdf/{i}.pdf", f"https://example.invalid/img/{i}.jpg"]
        rows.append(r)
    return rows


@pytest.fixture
def headerless_xlsx(tmp_path):
    p = tmp_path / "raw.xlsx"
    pd.DataFrame(make_rows()).to_excel(p, header=False, index=False)
    return str(p)


class FakeAnthropic:
    """Stand-in for anthropic.AsyncAnthropic: messages.create is async, no network."""
    calls = 0
    reply = {"affected_area_percent": 50, "crop_loss_percent": 40, "survey_date": "01092026",
             "surveyor_signed": True, "farmer_signed": True, "govt_signed": False,
             "form_remarks": "ok", "form_status": "correct", "confidence": 0.9,
             "field_photo_type": "standing crop", "farmer_photo_present": True, "loss_visible": True,
             "estimated_loss_pct": 40, "photo_date": "01092026", "photo_gps": None,
             "photo_quality": "good", "loss_matches_reported": True, "flags": []}
    tokens = (1000, 500)

    def __init__(self, *a, **k):
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kw):
        type(self).calls += 1
        await asyncio.sleep(0)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="```json\n" + json.dumps(type(self).reply) + "\n```")],
            usage=SimpleNamespace(input_tokens=self.tokens[0], output_tokens=self.tokens[1]))


@pytest.fixture
def fake_anthropic(monkeypatch):
    import anthropic
    FakeAnthropic.calls = 0
    FakeAnthropic.tokens = (1000, 500)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", FakeAnthropic)
    return FakeAnthropic
