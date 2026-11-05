"""The demo server's upload route: what it accepts, what it refuses, and that the analysis gets the file."""
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """scripts/demo_server.py on a free port, with a stand-in analyzer that records what it received."""
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_server", SCRIPTS / "demo_server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeAnalyzer:
        calls = []

        def analyze(self, path, sample_id, title, audio, source=None):
            self.calls.append((sf.info(str(path)).duration, title, audio))
            if title.startswith("fail"):
                raise RuntimeError("boom")
            return {"id": sample_id, "title": title, "audio": audio}

    module.Handler.analyzer = FakeAnalyzer()
    static = tmp_path_factory.mktemp("static")
    (static / "index.html").write_text("ok")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(module.Handler, directory=str(static)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", module, module.Handler.analyzer
    httpd.shutdown()


def post(url, body: bytes, name: str):
    req = urllib.request.Request(f"{url}/api/analyze", data=body, method="POST",
                                 headers={"X-Filename": urllib.request.quote(name)})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wav_bytes(tmp_path, seconds: float, suffix=".wav") -> bytes:
    path = tmp_path / f"a{suffix}"
    sf.write(path, np.zeros(int(22050 * seconds), dtype=np.float32), 22050, format=suffix[1:].upper())
    return path.read_bytes()


def test_health_lists_the_tested_formats(server):
    url, module, _ = server
    with urllib.request.urlopen(f"{url}/api/health") as r:
        body = json.loads(r.read())
    assert body["formats"] == [".wav", ".flac", ".ogg", ".mp3"] and body["maxBytes"] == module.MAX_BYTES


def test_a_supported_file_is_analysed_and_served_without_audio(server, tmp_path):
    url, _, analyzer = server
    status, body = post(url, wav_bytes(tmp_path, 2.0), "my song.wav")
    assert status == 200 and body["title"] == "my song.wav" and body["audio"] is None
    assert analyzer.calls[-1] == (2.0, "my song.wav", None)


@pytest.mark.parametrize("name", ["song.m4a", "notes.txt", "noext"])
def test_unsupported_types_are_refused(server, tmp_path, name):
    status, body = post(server[0], wav_bytes(tmp_path, 2.0), name)
    assert status == 400 and "unsupported file type" in body["error"]


def test_unreadable_audio_is_refused(server):
    status, body = post(server[0], b"not audio at all", "bad.wav")
    assert status == 400 and "could not be read as audio" in body["error"]


def test_too_short_is_refused(server, tmp_path):
    status, body = post(server[0], wav_bytes(tmp_path, 0.5), "short.wav")
    assert status == 400 and "1 s to" in body["error"]


def test_a_failed_analysis_is_reported(server, tmp_path):
    status, body = post(server[0], wav_bytes(tmp_path, 2.0), "fail.wav")
    assert status == 500 and "analysis failed" in body["error"]


def test_the_page_is_served(server):
    with urllib.request.urlopen(f"{server[0]}/index.html") as r:
        assert r.read() == b"ok"
