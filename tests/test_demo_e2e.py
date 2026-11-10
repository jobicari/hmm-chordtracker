"""The demo in a real browser (Chromium through Playwright): an imported file is played from exactly the
samples the server decodes, also after a seek, and the timeline's drags seek once, on release.

Needs Playwright (pip install playwright; playwright install chromium) and, for the MP3 case, lame; skipped
otherwise. The demo server runs V7.2 (tracked weights). The in-page checks are tests/e2e/demo_checks.js.
"""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

playwright = pytest.importorskip("playwright.sync_api")

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "tests" / "e2e" / "demo_checks.js"
FRAME_S = 2048 / 22050


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = free_port()
    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "CHORD_DEMO_WORKERS": "2", "CHORD_DEMO_MODEL": "v7_2"}
    proc = subprocess.Popen([sys.executable, str(REPO / "scripts" / "demo_server.py"), "--port", str(port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                if urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).status == 200:
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail("the demo server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(10)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """60 s of seeded noise (matches at one position only) as FLAC, and as VBR MP3 without its seek index."""
    d = tmp_path_factory.mktemp("audio")
    y = (0.2 * np.random.default_rng(0).standard_normal(44100 * 60)).astype(np.float32)
    sf.write(d / "noise.flac", y, 44100, subtype="PCM_16")
    out = {"flac": d / "noise.flac"}
    if shutil.which("lame"):
        sf.write(d / "noise.wav", y, 44100, subtype="PCM_16")
        subprocess.run(["lame", "--quiet", "-V2", "-t", str(d / "noise.wav"), str(d / "noise.mp3")], check=True)
        out["mp3"] = d / "noise.mp3"
    return out


@pytest.fixture()
def page(server):
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
        page = browser.new_page()
        page.goto(server)
        page.wait_for_function("view.mode === 'live' && !document.getElementById('file').disabled")
        page.add_script_tag(path=str(CHECKS))
        yield page
        browser.close()


def imported(page, path: Path):
    page.set_input_files("#file", str(path))
    page.wait_for_function("view.sample && view.sample.id === 'upload' && !document.getElementById('play').disabled",
                           timeout=60_000)
    return page


@pytest.mark.parametrize("kind", ["flac", "mp3"])
def test_after_a_seek_the_player_plays_what_the_server_decodes(page, files, kind):
    if kind not in files:
        pytest.skip("lame is not installed")
    imported(page, files[kind])
    for seek_to in (15.0, 40.0):
        r = page.evaluate(f"demoChecks.alignment({seek_to})")
        assert "error" not in r, r
        assert abs(r["sessionFrom"] - seek_to) < 1e-3
        assert abs(r["offset"]) < FRAME_S, f"{kind} at {seek_to} s: the player is {r['offset']:+.3f} s off"


def test_a_cancelled_drag_then_moves_do_not_seek(page, files):
    imported(page, files["flac"])
    page.evaluate("$('play').click()")
    r = page.evaluate("demoChecks.cancelledDrag()")
    assert r == {"sessionsOpened": 0, "moved": False}


def test_a_scrub_opens_one_session_on_release_and_chords_follow(page, files):
    imported(page, files["flac"])
    page.evaluate("$('play').click()")
    page.wait_for_timeout(1500)
    r = page.evaluate("demoChecks.scrub(5, 30)")
    assert r["sessionsDuring"] == 0 and r["sessionsOpened"] == 1 and abs(r["at"] - 30) < 0.2
    d = page.evaluate("demoChecks.decidedAfter(4)")
    assert d["playing"] and d["segments"] > 0
    assert d["knownEnd"] >= d["heard"] - 0.05                # every chord heard so far is decided (delay on)


def test_an_older_cached_live_js_gives_a_reload_message_not_a_type_error(page):
    """A browser that kept an older live.js (cached before the page files were sent no-cache) with this
    app.js: the import says to reload instead of failing with a TypeError (Safari: "Type error")."""
    r = page.evaluate("demoChecks.staleLiveJs()")
    assert "out of date" in r["status"] and "TypeError" not in r["status"]


@pytest.mark.parametrize("lead", [0, 0.1, 0.3, 0.5])
def test_every_look_ahead_shows_only_decided_chords_at_every_lead(page, files, lead):
    """With the delay on, the chords on screen (those of the time you hear, or `lead` s later) are already
    decided at every point; the lead adds itself to the delay."""
    imported(page, files["flac"])
    points = page.evaluate("view.sample.points.map((p) => p.id)")
    for point in points:
        r = page.evaluate(f"demoChecks.shownDecided('{point}', {lead}, 4)")
        assert r["playing"] and r["lead"] == lead
        assert r["knownEnd"] >= r["shown"] - 0.05, (point, r)
