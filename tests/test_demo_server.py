"""The demo server: the session protocol, its order, its limits, its origin check, the static page and the
settings, with the decoding in worker processes (pinned sessions, a worker that dies, the real chain's
labels unchanged). Workers here run FakeChain unless a test needs the model."""
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from tornado.httpclient import AsyncHTTPClient, HTTPClientError, HTTPRequest
from tornado.websocket import websocket_connect

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SR, HOP = 22050, 2048
POINTS = ["0.28 s", "0.46 s", "0.65 s"]
CRASH = -32768                       # a block starting with this sample makes the fake worker exit


class FakeSession:
    """One label per 1,000 samples at every point, the label being the worker's pid; finish() adds one more."""

    def __init__(self):
        self.labels = {p: [] for p in POINTS}
        self.samples = 0

    def step(self, block):
        if len(block) and block[0] == CRASH / 32768.0:
            os._exit(3)
        before = self.samples // 1000
        self.samples += len(block)
        for _ in range(self.samples // 1000 - before):
            for labels in self.labels.values():
                labels.append(os.getpid())

    def finish(self):
        for labels in self.labels.values():
            labels.append(0)


class FakeChain:
    def new_session(self):
        return FakeSession()

    def info(self):
        return {"points": [{"id": p} for p in POINTS], "defaultPoint": POINTS[-1], "framewiseDecisionDelaySeconds": 0.3,
                "metadata": {}, "labels": [f"C{i}" for i in range(24)], "sampleRate": SR}


@pytest.fixture(scope="module")
def server_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_server", SCRIPTS / "demo_server.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["demo_server"] = module                 # the workers unpickle demo_server.Chain by name
    spec.loader.exec_module(module)
    return module


def run(server_module, tmp_path, test, env=None, workers=1, chain=FakeChain):
    """Start the workers and the app on a free port, run the async test against it, stop."""
    (tmp_path / "index.html").write_text("ok")
    (tmp_path / "README.md").write_text("not part of the page")
    (tmp_path / "samples").mkdir(exist_ok=True)
    (tmp_path / "samples" / "index.json").write_text('{"samples": []}')
    settings = server_module.parse_args([], env={"CHORD_DEMO_WORKERS": str(workers), **(env or {})})
    server_module.LiveHandler.sessions = 0

    async def main():
        pool = server_module.Workers(chain, settings.workers)
        await pool.start()
        httpd = server_module.make_app(pool, tmp_path, settings).listen(0, address="127.0.0.1")
        port = next(iter(httpd._sockets.values())).getsockname()[1]
        try:
            await test(f"127.0.0.1:{port}", pool)
        finally:
            httpd.stop()
            await pool.stop()

    asyncio.run(main())


async def messages_until(ws, kind):
    out = []
    while True:
        m = json.loads(await ws.read_message())
        out.append(m)
        if m["type"] in (kind, "error"):
            return out


async def session(host, pcm, block=700, origin=None):
    """Stream pcm (int16) in blocks, end, and return every point's labels (checked contiguous and in order)."""
    headers = {} if origin is None else {"Origin": origin}
    ws = await websocket_connect(HTTPRequest(f"ws://{host}/api/live", headers=headers))
    await ws.write_message(json.dumps({"type": "start"}))
    ready = json.loads(await ws.read_message())
    assert ready["type"] == "ready", ready
    for i in range(0, len(pcm), block):
        await ws.write_message(pcm[i:i + block].astype("<i2").tobytes(), binary=True)
    await ws.write_message(json.dumps({"type": "end"}))
    received = {}
    for m in await messages_until(ws, "done"):
        assert m["type"] != "error", m
        for point, v in m.get("points", {}).items():
            assert v["from"] == len(received.setdefault(point, []))
            received[point] += v["labels"]
    ws.close()
    return ready, received


def test_health_and_page(server_module, tmp_path):
    async def test(host, pool):
        client = AsyncHTTPClient()
        health = json.loads((await client.fetch(f"http://{host}/api/health")).body)
        assert health["live"] and [p["id"] for p in health["points"]] == POINTS
        assert health["workers"] == {"alive": 1, "count": 1} and health["maxSessions"] == 8
        assert json.loads((await client.fetch(f"http://{host}/healthz")).body) == health
        assert (await client.fetch(f"http://{host}/")).body == b"ok"
        assert json.loads((await client.fetch(f"http://{host}/samples/index.json")).body) == {"samples": []}
        for path in ("README.md", "samples/../README.md"):
            with pytest.raises(HTTPClientError) as refused:
                await client.fetch(f"http://{host}/{path}")
            assert refused.value.code in (403, 404)
    run(server_module, tmp_path, test)


def test_page_files_are_revalidated_on_every_load(server_module, tmp_path):
    """A new image's page reaches browsers that loaded the old one: no-cache, and 304 while unchanged."""
    async def test(host, pool):
        client = AsyncHTTPClient()
        for path in ("", "index.html", "samples/index.json"):
            first = await client.fetch(f"http://{host}/{path}")
            assert first.headers["Cache-Control"] == "no-cache" and first.headers["Etag"]
            again = await client.fetch(f"http://{host}/{path}", headers={"If-None-Match": first.headers["Etag"]},
                                       raise_error=False)
            assert again.code == 304
    run(server_module, tmp_path, test)


def test_a_session_streams_labels_in_order_and_flushes_at_the_end(server_module, tmp_path):
    async def test(host, pool):
        ready, received = await session(host, np.zeros(5500))
        assert ready["sampleRate"] == SR and len(ready["labels"]) == 24
        worker = pool.slots[0].process.pid
        assert received == {p: [worker] * 5 + [0] for p in POINTS}
    run(server_module, tmp_path, test)


def test_sessions_are_pinned_to_one_worker_each_spread_over_the_workers(server_module, tmp_path):
    async def test(host, pool):
        results = await asyncio.gather(*(session(host, np.zeros(20000)) for _ in range(2)))
        pids = [set(received[POINTS[0]][:-1]) for _, received in results]
        assert all(len(p) == 1 for p in pids)                                  # every block in one process
        assert pids[0] | pids[1] == {s.process.pid for s in pool.slots}          # one session on each worker
    run(server_module, tmp_path, test, workers=2)


def test_a_worker_that_dies_ends_its_sessions_and_is_restarted(server_module, tmp_path):
    async def test(host, pool):
        first = pool.slots[0].process.pid
        ws = await websocket_connect(f"ws://{host}/api/live")
        await ws.write_message(json.dumps({"type": "start"}))
        assert json.loads(await ws.read_message())["type"] == "ready"
        await ws.write_message(np.full(100, CRASH, "<i2").tobytes(), binary=True)
        m = json.loads(await ws.read_message())
        assert m["type"] == "error" and "stopped" in m["message"]
        for _ in range(600):                                                    # restarted (new process loaded)
            if pool.alive == 1:
                break
            await asyncio.sleep(0.05)
        assert pool.alive == 1 and pool.slots[0].process.pid != first
        _, received = await session(host, np.zeros(3000))
        assert received[POINTS[0]] == [pool.slots[0].process.pid] * 3 + [0]
    run(server_module, tmp_path, test)


def test_audio_before_start_is_refused(server_module, tmp_path):
    async def test(host, pool):
        ws = await websocket_connect(f"ws://{host}/api/live")
        await ws.write_message(np.zeros(100, "<i2").tobytes(), binary=True)
        m = json.loads(await ws.read_message())
        assert m["type"] == "error" and "start" in m["message"]
    run(server_module, tmp_path, test)


def test_sessions_and_length_are_limited(server_module, tmp_path):
    async def test(host, pool):
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
    run(server_module, tmp_path, test, env={"CHORD_DEMO_MAX_SESSIONS": "1", "CHORD_DEMO_MAX_SECONDS": "0.1"})


async def live_with_origin(host, origin):
    """Open the live route as a browser page from `origin` would (None: no Origin, as a script) and start."""
    headers = {} if origin is None else {"Origin": origin}
    ws = await websocket_connect(HTTPRequest(f"ws://{host}/api/live", headers=headers))
    await ws.write_message(json.dumps({"type": "start"}))
    m = json.loads(await ws.read_message())
    ws.close()
    return m["type"]


def test_the_live_route_accepts_its_own_page_and_allowed_origins_only(server_module, tmp_path):
    async def test(host, pool):
        port = host.split(":")[1]
        assert await live_with_origin(host, f"http://{host}") == "ready"                 # the server's own page
        assert await live_with_origin(host, None) == "ready"                             # not a browser
        assert await live_with_origin(host, "https://Pages.Example.org/") == "ready"     # allowed
        for origin in ("https://evil.example.com", f"http://localhost.evil.example.com:{port}",
                       "http://pages.example.org"):                                     # another scheme
            with pytest.raises(HTTPClientError) as refused:
                await live_with_origin(host, origin)
            assert refused.value.code == 403
    run(server_module, tmp_path, test, env={"CHORD_DEMO_ALLOWED_ORIGINS": "https://pages.example.org"})


def test_settings_come_from_flags_then_the_environment(server_module, monkeypatch):
    parse = server_module.parse_args
    monkeypatch.setattr(server_module, "available_cpus", lambda: 12)
    default = parse([], env={})
    assert (default.host, default.port, default.allowed_origins) == ("127.0.0.1", 8000, [])
    assert (default.workers, default.max_sessions, default.max_seconds, default.live_margin) == (8, 8, 600.0, 0.15)
    assert default.model == "v14" and default.silence_dbfs == float("-inf")         # v14: its own N, no gate
    assert parse([], env={"CHORD_DEMO_MODEL": "v7_2"}).silence_dbfs == -55.0        # v7_2: the gate
    assert parse(["--model", "v7_2"], env={"CHORD_DEMO_MODEL": "v14"}).model == "v7_2"
    with pytest.raises(SystemExit):
        parse([], env={"CHORD_DEMO_MODEL": "v99"})
    monkeypatch.setattr(server_module, "available_cpus", lambda: 2)
    assert parse([], env={}).workers == 2                                      # the CPUs, at most 8 by default
    assert parse([], env={"PORT": "9000"}).port == 9000
    assert parse([], env={"PORT": "9000", "CHORD_DEMO_PORT": "9100"}).port == 9100
    env = {"CHORD_DEMO_HOST": "0.0.0.0", "CHORD_DEMO_PORT": "9100", "CHORD_DEMO_WORKERS": "12",
           "CHORD_DEMO_MAX_SESSIONS": "4", "CHORD_DEMO_MAX_SECONDS": "60", "CHORD_DEMO_LIVE_MARGIN": "0.4",
           "CHORD_DEMO_SILENCE_DBFS": "-50", "CHORD_DEMO_ALLOWED_ORIGINS": " https://a.example.org, ,https://b.example.org "}
    s = parse([], env=env)
    assert (s.host, s.port, s.workers, s.max_sessions, s.max_seconds, s.live_margin) == ("0.0.0.0", 9100, 12, 4, 60.0, 0.4)
    assert s.allowed_origins == ["https://a.example.org", "https://b.example.org"] and s.silence_dbfs == -50.0
    assert parse(["--silence-dbfs", "-60"], env=env).silence_dbfs == -60.0
    flags = parse(["--host", "::", "--port", "8010", "--allowed-origins", "https://c.example.org", "--workers", "3"],
                  env=env)
    assert (flags.host, flags.port, flags.allowed_origins, flags.workers) == ("::", 8010, ["https://c.example.org"], 3)
    with pytest.raises(SystemExit):
        parse([], env={"CHORD_DEMO_WORKERS": "0"})


def test_health_reports_the_settings_in_use(server_module, tmp_path):
    async def test(host, pool):
        health = json.loads((await AsyncHTTPClient().fetch(f"http://{host}/healthz")).body)
        assert (health["maxSessions"], health["maxSeconds"], health["liveMarginSeconds"]) == (3, 120.0, 0.4)
        assert health["silenceDbfs"] == -50.0 and health["model"] == "v14"
        assert health["workers"] == {"alive": 2, "count": 2}
    run(server_module, tmp_path, test, workers=2,
        env={"CHORD_DEMO_MAX_SESSIONS": "3", "CHORD_DEMO_MAX_SECONDS": "120", "CHORD_DEMO_LIVE_MARGIN": "0.4",
             "CHORD_DEMO_SILENCE_DBFS": "-50"})


def test_the_silence_gate_can_be_turned_off(server_module):
    settings = server_module.parse_args([], env={"CHORD_DEMO_SILENCE_DBFS": "-inf"})
    class NoWorkers:
        info, alive, n = None, 0, 1
    assert server_module.health(NoWorkers(), settings)["silenceDbfs"] is None      # JSON has no -inf


def test_the_servers_lists_are_the_models(server_module):
    """The server process does not import the model code: its copies of the names and defaults must agree."""
    sys.path.insert(0, str(SCRIPTS))
    import demo_export
    from chordtracker import demo
    assert set(server_module.MODELS) == set(demo.MODELS) and server_module.DEFAULT_MODEL == demo.DEFAULT_MODEL
    assert server_module.DEFAULT_SILENCE_DBFS == demo_export.DEFAULT_SILENCE_DBFS


V14_MODEL = SCRIPTS.parent / "results" / "v14" / "n_seed0" / "model.pt"
needs_v14 = pytest.mark.skipif(not V14_MODEL.exists(), reason="results/v14/n_seed0/model.pt is not in git: copy it in")


@pytest.mark.parametrize("model", [pytest.param("v14", marks=needs_v14), "v7_2"])
def test_the_real_chain_gives_the_same_labels_through_a_worker(server_module, tmp_path, model):
    """The process path changes nothing: the labels of a worker session are those of the chain run here."""
    import functools
    from test_features_v7_1 import synthetic
    pcm = np.clip(np.round(synthetic(seconds=12.0, cents=15.0) * 32767), -32768, 32767).astype("<i2")
    chain = server_module.Chain(model)
    here = chain.new_session()
    for i in range(0, len(pcm), 3000):
        here.step(pcm[i:i + 3000].astype(np.float32) / 32768.0)
    here.finish()

    async def test(host, pool):
        ready, received = await session(host, pcm, block=3000)
        assert len(ready["labels"]) == {"v14": 61, "v7_2": 25}[model] and ready["labels"][-1] == "N"
        assert received == {p: [int(x) for x in labels] for p, labels in here.labels.items()}
        health = json.loads((await AsyncHTTPClient().fetch(f"http://{host}/healthz")).body)
        assert health["metadata"]["pipeline"]["network"]["name"] == model
    run(server_module, tmp_path, test, chain=functools.partial(server_module.Chain, model),
        env={"CHORD_DEMO_MODEL": model})


def triads(chords, seconds, cents=0.0):
    """Root-position triads (Harte "C:maj", "A:min") in octave 3 with 6 harmonics each, `seconds` apiece."""
    import librosa
    t = np.arange(int(seconds * SR)) / SR
    out = []
    for chord in chords:
        root, quality = chord.split(":")
        base = librosa.note_to_midi(root + "3")
        y = np.zeros(len(t))
        for midi in (base, base + (4 if quality == "maj" else 3), base + 7):
            for h in range(1, 7):
                y += np.sin(2 * np.pi * h * librosa.midi_to_hz(midi + cents / 100) * t) / h
        out.append(0.05 * y * np.minimum(1, t / 0.01))
    return np.concatenate(out).astype(np.float32)


def test_a_silent_intro_is_no_chord_then_the_recordings_chords(server_module, tmp_path):
    """3.7 s of digital silence (40 frames), then 12 s of triads 15 cents sharp, through a worker of V7.2's
    chain with the silence gate: N (label 24) on the intro at every point, a chord from the onset on, the
    tuning of the music alone, and its labels once the running level has forgotten the intro (its silent
    columns lower the 99th percentile for a few seconds)."""
    import functools
    music = triads(["C:maj", "A:min", "F:maj", "G:maj", "E:min", "D:min"], 2.0, cents=15.0)
    intro = 40
    y = np.concatenate([np.zeros(intro * HOP, np.float32), music])
    pcm = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2")
    chain = server_module.Chain("v7_2", -55.0)

    def run_here(samples):
        a = chain.new_session()
        for i in range(0, len(samples), 3000):
            a.step(samples[i:i + 3000].astype(np.float32) / 32768.0)
        a.finish()
        return a
    here, alone = run_here(pcm), run_here(pcm[intro * HOP:])
    assert here.features.tuned.tuning == alone.features.tuned.tuning
    for point, labels in here.labels.items():
        assert set(labels[:intro]) == {24} and 24 not in labels[intro:]
        settled = 43                                      # 4 s after the onset
        assert np.mean(np.array(labels[intro + settled:]) == np.array(alone.labels[point][settled:])) > 0.9

    async def test(host, pool):
        ready, received = await session(host, pcm, block=3000)
        assert len(ready["labels"]) == 25 and ready["labels"][24] == "N"
        assert received == {p: [int(x) for x in labels] for p, labels in here.labels.items()}
    run(server_module, tmp_path, test, chain=functools.partial(server_module.Chain, "v7_2", -55.0))


@needs_v14
def test_v14_says_no_chord_itself_on_a_silent_intro(server_module):
    """V14 without the gate: its own N on 3.7 s of digital silence at every point, chords once the music
    has started (at most 2 frames of N at the onset), and the tuning of the music alone (the tuning
    seconds start at the first audible frame with the gate off too)."""
    music = triads(["C:maj", "A:min", "F:maj", "G:maj", "E:min", "D:min"], 2.0, cents=15.0)
    intro = 40
    chain = server_module.Chain("v14")
    assert chain.silence_dbfs == float("-inf")

    def run_here(samples):
        a = chain.new_session()
        for i in range(0, len(samples), 3000):
            a.step(samples[i:i + 3000])
        a.finish()
        return a
    here, alone = run_here(np.concatenate([np.zeros(intro * HOP, np.float32), music])), run_here(music)
    assert here.silent == [] and here.features.tuned.tuning == alone.features.tuned.tuning
    for labels in here.labels.values():
        assert set(labels[:intro]) == {60} and sum(x == 60 for x in labels[intro:]) <= 2


@pytest.mark.parametrize("model", [pytest.param("v14", marks=needs_v14), "v7_2"])
def test_the_prepared_sample_is_the_live_session(server_module, tmp_path, model):
    """demo_export's JSON of a file has, at every point, the labels a live session gives for the same PCM."""
    import soundfile as sf
    sys.path.insert(0, str(SCRIPTS))
    import demo_export
    from chordtracker import demo
    y = np.concatenate([np.zeros(10 * HOP, np.float32), triads(["G:maj", "E:min", "C:maj"], 2.0)])
    sf.write(tmp_path / "x.wav", y, SR, subtype="FLOAT")
    analyzer = demo_export.Analyzer(model)
    out = analyzer.analyze(tmp_path / "x.wav", "x", "X", "x.wav")
    chain = server_module.Chain(model)
    live = chain.new_session()
    pcm = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2")
    for i in range(0, len(pcm), 3000):
        live.step(pcm[i:i + 3000].astype(np.float32) / 32768.0)
    live.finish()
    times = np.arange(len(live.labels["0.28 s"])) * HOP / SR
    for p in out["points"]:
        assert p["segments"] == demo.path_segments(np.array(live.labels[p["id"]]), chain.labels, times, out["duration"])
    assert out["metadata"]["pipeline"]["network"]["name"] == model
