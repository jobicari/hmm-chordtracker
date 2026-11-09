"""The demo server's live route: the session protocol, its order, its limits, and the static page."""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tornado.httpclient import AsyncHTTPClient
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


def run(server_module, tmp_path, test):
    """Start the app on a free port with the fake chain, run the async test against it, stop."""
    (tmp_path / "index.html").write_text("ok")

    async def main():
        app = server_module.make_app(FakeChain(), tmp_path)
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
        assert (await client.fetch(f"http://{host}/")).body == b"ok"
    run(server_module, tmp_path, test)


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
