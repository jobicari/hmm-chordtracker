"""The demo server's live route: the session protocol, its order, its limits, its origin check, the static
page and the settings."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPRequest
from tornado.websocket import websocket_connect

from chordtracker import demo
from chordtracker.play_along import V7_2_POINTS as OPERATING_POINTS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SR, HOP = 22050, 2048


@pytest.fixture(scope="module")
def server_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_server", SCRIPTS / "demo_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSession:
    """One label per 1,000 samples at every point; finish() adds one more (the flush)."""

    def __init__(self):
        self.labels = {p: [] for p in OPERATING_POINTS}
        self.samples = 0

    def step(self, block):
        before = self.samples // 1000
        self.samples += len(block)
        for _ in range(self.samples // 1000 - before):
            for labels in self.labels.values():
                labels.append(len(labels) % 24)

    def finish(self):
        for labels in self.labels.values():
            labels.append(0)


class FakeChain:
    sr, hop = SR, HOP

    def new_session(self):
        return FakeSession()

    def health(self):
        return {"live": True, "points": demo.point_info(SR, HOP), "maxSeconds": 600.0}


def run(server_module, tmp_path, test, allowed_origins=()):
    """Start the app on a free port with the fake chain, run the async test against it, stop."""
    (tmp_path / "index.html").write_text("ok")
    (tmp_path / "README.md").write_text("not part of the page")
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "index.json").write_text('{"samples": []}')

    async def main():
        app = server_module.make_app(FakeChain(), tmp_path, allowed_origins)
        httpd = app.listen(0, address="127.0.0.1")
        port = next(iter(httpd._sockets.values())).getsockname()[1]
        try:
            await test(f"127.0.0.1:{port}")
        finally:
            httpd.stop()

    asyncio.run(main())


async def messages_until(ws, kind):
    out = []
    while True:
        m = json.loads(await ws.read_message())
        out.append(m)
        if m["type"] in (kind, "error"):
            return out


def test_health_and_page(server_module, tmp_path):
    async def test(host):
        client = AsyncHTTPClient()
        health = json.loads((await client.fetch(f"http://{host}/api/health")).body)
        assert health["live"] and [p["id"] for p in health["points"]] == list(OPERATING_POINTS)
        assert json.loads((await client.fetch(f"http://{host}/healthz")).body) == health
        assert (await client.fetch(f"http://{host}/")).body == b"ok"
        assert (await client.fetch(f"http://{host}/index.html")).body == b"ok"
        assert json.loads((await client.fetch(f"http://{host}/samples/index.json")).body) == {"samples": []}
        for path in ("README.md", "samples/../README.md"):
            with pytest.raises(HTTPClientError) as refused:
                await client.fetch(f"http://{host}/{path}")
            assert refused.value.code in (403, 404)
    run(server_module, tmp_path, test)


async def live_with_origin(host, origin):
    """Open the live route as a browser page from `origin` would (None: no Origin, as a script) and start."""
    headers = {} if origin is None else {"Origin": origin}
    ws = await websocket_connect(HTTPRequest(f"ws://{host}/api/live", headers=headers))
    await ws.write_message(json.dumps({"type": "start"}))
    m = json.loads(await ws.read_message())
    ws.close()
    return m["type"]


def test_the_live_route_accepts_its_own_page_and_allowed_origins_only(server_module, tmp_path):
    async def test(host):
        port = host.split(":")[1]
        assert await live_with_origin(host, f"http://{host}") == "ready"                 # the server's own page
        assert await live_with_origin(host, None) == "ready"                             # not a browser
        assert await live_with_origin(host, "https://Pages.Example.org/") == "ready"     # allowed
        for origin in ("https://evil.example.com", f"http://localhost.evil.example.com:{port}",
                       "http://pages.example.org"):                                     # another scheme
            with pytest.raises(HTTPClientError) as refused:
                await live_with_origin(host, origin)
            assert refused.value.code == 403
    run(server_module, tmp_path, test, allowed_origins=["https://pages.example.org"])


def test_health_tells_the_page_its_live_margin(server_module, monkeypatch):
    monkeypatch.setattr(server_module, "LIVE_MARGIN", 0.4)
    chain = type("C", (), {"sr": SR, "hop": HOP, "metadata": {}})()
    health = server_module.Chain.health(chain)
    assert health["liveMarginSeconds"] == 0.4 and health["maxSeconds"] == server_module.MAX_SECONDS


def test_settings_come_from_flags_then_the_environment(server_module):
    parse = server_module.parse_args
    default = parse([], env={})
    assert (default.host, default.port, default.allowed_origins) == ("127.0.0.1", 8000, [])
    assert parse([], env={"PORT": "9000"}).port == 9000
    assert parse([], env={"PORT": "9000", "CHORD_DEMO_PORT": "9100"}).port == 9100
    env = {"CHORD_DEMO_HOST": "0.0.0.0", "CHORD_DEMO_PORT": "9100",
           "CHORD_DEMO_ALLOWED_ORIGINS": " https://a.example.org, ,https://b.example.org "}
    from_env = parse([], env=env)
    assert (from_env.host, from_env.port) == ("0.0.0.0", 9100)
    assert from_env.allowed_origins == ["https://a.example.org", "https://b.example.org"]
    flags = parse(["--host", "::", "--port", "8010", "--allowed-origins", "https://c.example.org"], env=env)
    assert (flags.host, flags.port, flags.allowed_origins) == ("::", 8010, ["https://c.example.org"])


def test_a_session_streams_labels_in_order_and_flushes_at_the_end(server_module, tmp_path):
    async def test(host):
        ws = await websocket_connect(f"ws://{host}/api/live")
        await ws.write_message(json.dumps({"type": "start"}))
        ready = json.loads(await ws.read_message())
        assert ready["type"] == "ready" and ready["sampleRate"] == SR and len(ready["labels"]) == 24
        pcm = (np.zeros(5500)).astype("<i2")
        for i in range(0, len(pcm), 700):                       # any block size
            await ws.write_message(pcm[i:i + 700].tobytes(), binary=True)
        await ws.write_message(json.dumps({"type": "end"}))
        received = {p: [] for p in OPERATING_POINTS}
        for m in await messages_until(ws, "done"):
            for point, v in m.get("points", {}).items():
                assert v["from"] == len(received[point])          # contiguous, in order
                received[point] += v["labels"]
        assert all(labels == [0, 1, 2, 3, 4, 0] for labels in received.values())
        ws.close()
    run(server_module, tmp_path, test)


def test_audio_before_start_is_refused(server_module, tmp_path):
    async def test(host):
        ws = await websocket_connect(f"ws://{host}/api/live")
        await ws.write_message(np.zeros(100, "<i2").tobytes(), binary=True)
        m = json.loads(await ws.read_message())
        assert m["type"] == "error" and "start" in m["message"]
    run(server_module, tmp_path, test)


def test_sessions_and_length_are_limited(server_module, tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "MAX_SESSIONS", 1)
    monkeypatch.setattr(server_module, "MAX_SECONDS", 0.1)
    monkeypatch.setattr(server_module.LiveHandler, "sessions", 0)      # sessions of earlier tests

    async def test(host):
        first = await websocket_connect(f"ws://{host}/api/live")
        await first.write_message(json.dumps({"type": "start"}))
        assert json.loads(await first.read_message())["type"] == "ready"
        second = await websocket_connect(f"ws://{host}/api/live")
        await second.write_message(json.dumps({"type": "start"}))
        m = json.loads(await second.read_message())
        assert m["type"] == "error" and "busy" in m["message"]
        await first.write_message(np.zeros(int(0.2 * SR), "<i2").tobytes(), binary=True)
        m = json.loads(await first.read_message())
        assert m["type"] == "error" and "at most" in m["message"]
    run(server_module, tmp_path, test)
