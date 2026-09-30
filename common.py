"""Shared helpers: cached downloader, cost tracker, JSON parsing, geo maths."""
import asyncio
import hashlib
import json
import math
import re
from pathlib import Path

import httpx

import config as C


class CostTracker:
    def __init__(self, cap=None, spent=0.0):
        self.cap, self.spent = cap, spent

    def add(self, model, usage):
        pin, pout = C.MODEL_PRICING.get(model, C.MODEL_PRICING["default"])
        self.spent += (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6

    @property
    def exceeded(self):
        return self.cap is not None and self.spent >= self.cap


class Unavailable(Exception):
    pass


async def fetch(http: httpx.AsyncClient, url: str, cache_dir: str, timeout: float) -> Path:
    """Cache -> direct GET -> GET with Referer. Raises Unavailable on failure."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / hashlib.sha1(url.encode()).hexdigest()
    if path.exists() and path.stat().st_size:
        return path
    last = None
    for headers in ({}, {"Referer": C.REFERER}):
        try:
            r = await http.get(url, headers=headers, timeout=timeout, follow_redirects=True)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return path
            last = f"HTTP {r.status_code}"
            if r.status_code not in (401, 403):
                break
        except httpx.HTTPError as e:
            last = type(e).__name__
    raise Unavailable(last or "unavailable")


def image_media_type(b: bytes) -> str:
    if b[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "image/webp"
    return ""


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


async def call_claude(client, tracker: CostTracker, model: str, content: list, retries=4, max_tokens=800):
    for attempt in range(retries):
        try:
            resp = await client.messages.create(model=model, max_tokens=max_tokens,
                                                messages=[{"role": "user", "content": content}])
            tracker.add(model, resp.usage)
            return parse_json("".join(b.text for b in resp.content if b.type == "text"))
        except json.JSONDecodeError:
            if attempt == retries - 1:
                raise
        except Exception as e:  # rate limit / overload / transient
            status = getattr(e, "status_code", None)
            if attempt == retries - 1 or status in (400, 401, 403):
                raise
            await asyncio.sleep(2 ** (attempt + 1))


def haversine_m(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def num(v):
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
