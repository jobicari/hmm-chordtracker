"""Demo website with live decoding: serves demo/ and decodes the audio the page streams to it, as it plays.

    python scripts/demo_server.py            # then open http://localhost:8000
    python scripts/demo_server.py --port 8010
    python scripts/demo_server.py --host 0.0.0.0     # reachable from other machines (the Docker image)

The page decodes the recording (a prepared sample or the listener's own file), resamples it to the
model's rate and streams it at playback speed; this server runs the listening prototype's live chain on
it (play_along.Analysis: block-by-block features, the causal CRNN, the three operating points' decoders)
and sends every chord back as soon as it is decided. Routes:

    GET  /api/health    {"live": true, "points": [...], "metadata": {...}, "maxSeconds": ...}: what a
                        page needs to decode a file of its own (the same fields as a sample's JSON)
    GET  /healthz       the same, for a container's health check
    WS   /api/live      one session per connection:
                          client {"type": "start"}         -> {"type": "ready", "labels": [...], "sampleRate": ...}
                          client binary: 16-bit mono PCM at sampleRate, any block size
                                                           -> {"type": "labels", "points": {id: {"from": k, "labels": [...]}}}
                          client {"type": "end"}           -> the last labels (decoders flushed), then {"type": "done"}
                          errors                           -> {"type": "error", "message": ...}, then the server closes

Each session has its own features, network state and decoders; the model is shared. The work of a block
runs in a thread pool, one block at a time per session (tornado delivers a connection's messages in
order). At most MAX_SESSIONS sessions run at once and each stops after MAX_SECONDS of audio. Only the
page's own files and samples are served.

Settings: --host (default 127.0.0.1, this machine only), --port (default 8000), --allowed-origins; each
also from the environment: CHORD_DEMO_HOST, CHORD_DEMO_PORT (else PORT, as many hosts set it),
CHORD_DEMO_ALLOWED_ORIGINS (comma-separated, e.g. "https://example.github.io"). The live route accepts a
page from the server's own origin, and from these. CHORD_DEMO_MAX_SESSIONS and CHORD_DEMO_MAX_SECONDS
override the limits.
"""
import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
import tornado.web
import tornado.websocket

from chordtracker import demo, paths
from chordtracker.learned_emission import LABELS
from chordtracker.play_along import V7_2_POINTS as OPERATING_POINTS, Analysis

sys.path.insert(0, str(Path(__file__).parent))
import demo_export  # noqa: E402

MAX_SESSIONS = int(os.environ.get("CHORD_DEMO_MAX_SESSIONS", 8))
MAX_SECONDS = float(os.environ.get("CHORD_DEMO_MAX_SECONDS", 600.0))
PAGE_FILES = r"/(|index\.html|style\.css|app\.js|live\.js|samples/[^/]+)"     # not README.md


def origin_key(origin: str) -> str:
    """"https://Example.org/" -> "https://example.org": an origin as browsers send it, for comparison."""
    parts = urlsplit(origin.strip())
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


class Chain:
    """The model and its settings, loaded once; new_session() -> a fresh play_along.Analysis."""

    def __init__(self):
        self.analyzer = demo_export.Analyzer()
        a = self.analyzer
        self.sr, self.hop = a.cfg["audio"]["sr"], a.cfg["cqt"]["hop_length"]
        self.metadata = {"sampleRate": self.sr, "hopLength": self.hop, "featureName": demo_export.FEATURE,
                         "model": str(a.run.relative_to(paths.REPO_ROOT)),
                         "inputLookaheadFrames": demo_export.INPUT_LOOKAHEAD,
                         "frameSeconds": round(self.hop / self.sr, 6),
                         "pipeline": demo_export.pipeline_settings(a.cfg, a.run)}

    def new_session(self):
        return Analysis(self.analyzer.cfg, self.analyzer.model, self.analyzer.prior, self.analyzer.Q, OPERATING_POINTS)

    def health(self) -> dict:
        return {"live": True, "points": demo.point_info(self.sr, self.hop), "defaultPoint": demo.DEFAULT_POINT,
                "framewiseDecisionDelaySeconds": demo.framewise_decision_delay(self.sr, self.hop),
                "metadata": self.metadata, "maxSeconds": MAX_SECONDS}


class HealthHandler(tornado.web.RequestHandler):
    def initialize(self, chain):
        self.chain = chain

    def get(self):
        self.set_header("Cache-Control", "no-store")
        self.write(self.chain.health())


class LiveHandler(tornado.websocket.WebSocketHandler):
    """One live session: PCM blocks in, decided chords out."""
    sessions = 0

    def initialize(self, chain, pool, allowed_origins):
        self.chain, self.pool, self.allowed_origins = chain, pool, allowed_origins
        self.analysis, self.sent, self.samples, self.counted = None, {}, 0, False

    def check_origin(self, origin):
        """The server's own page (tornado's same-origin check), or a page from an allowed origin."""
        return origin_key(origin) in self.allowed_origins or super().check_origin(origin)

    async def on_message(self, message):
        try:
            if isinstance(message, bytes):
                await self.on_audio(message)
            else:
                await self.on_command(json.loads(message))
        except tornado.websocket.WebSocketClosedError:
            pass

    async def on_command(self, msg: dict):
        kind = msg.get("type")
        if kind == "start":
            if self.analysis is not None:
                return self.fail("The session has already started.")
            if LiveHandler.sessions >= MAX_SESSIONS:
                return self.fail("The demo server is busy; try again in a moment.")
            LiveHandler.sessions += 1
            self.counted = True
            self.analysis = self.chain.new_session()
            self.sent = {p: 0 for p in OPERATING_POINTS}
            await self.write_message({"type": "ready", "labels": list(LABELS), "sampleRate": self.chain.sr})
        elif kind == "end":
            if self.analysis is None:
                return self.fail("No session.")
            await asyncio.get_running_loop().run_in_executor(self.pool, self.analysis.finish)
            await self.send_new()
            await self.write_message({"type": "done"})
        else:
            self.fail(f"Unknown message {kind!r}.")

    async def on_audio(self, data: bytes):
        if self.analysis is None:
            return self.fail("Send {\"type\": \"start\"} first.")
        block = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.samples += len(block)
        if self.samples > MAX_SECONDS * self.chain.sr:
            return self.fail(f"The demo decodes at most {MAX_SECONDS / 60:.0f} min per session.")
        await asyncio.get_running_loop().run_in_executor(self.pool, self.analysis.step, block)
        await self.send_new()

    async def send_new(self):
        new = {}
        for point, labels in self.analysis.labels.items():
            if len(labels) > self.sent[point]:
                new[point] = {"from": self.sent[point], "labels": labels[self.sent[point]:]}
                self.sent[point] = len(labels)
        if new:
            await self.write_message({"type": "labels", "points": new})

    def fail(self, message: str):
        self.write_message({"type": "error", "message": message})
        self.close()

    def on_close(self):
        if self.counted:
            LiveHandler.sessions -= 1
            self.counted = False


def make_app(chain, static_dir: Path, allowed_origins=()) -> tornado.web.Application:
    pool = ThreadPoolExecutor(max_workers=MAX_SESSIONS)
    live = {"chain": chain, "pool": pool, "allowed_origins": {origin_key(o) for o in allowed_origins}}
    return tornado.web.Application([
        (r"/api/health", HealthHandler, {"chain": chain}),
        (r"/healthz", HealthHandler, {"chain": chain}),
        (r"/api/live", LiveHandler, live),
        (PAGE_FILES, tornado.web.StaticFileHandler, {"path": str(static_dir), "default_filename": "index.html"}),
    ], websocket_max_message_size=2 ** 20)


def parse_args(argv=None, env=os.environ) -> argparse.Namespace:
    """The command line, with defaults from the environment (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=env.get("CHORD_DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env.get("CHORD_DEMO_PORT", env.get("PORT", 8000))))
    parser.add_argument("--allowed-origins", default=env.get("CHORD_DEMO_ALLOWED_ORIGINS", ""),
                        help="comma-separated origins whose pages may use the live route")
    args = parser.parse_args(argv)
    args.allowed_origins = [o for o in (s.strip() for s in args.allowed_origins.split(",")) if o]
    return args


async def serve(host: str, port: int, allowed_origins) -> None:
    torch.set_num_threads(1)                         # one block at a time per session; sessions in parallel
    app = make_app(Chain(), paths.REPO_ROOT / "demo", allowed_origins)
    app.listen(port, address=host)
    print(f"Demo: http://localhost:{port} (listening on {host})", flush=True)
    await asyncio.Event().wait()


def main() -> None:
    args = parse_args()
    asyncio.run(serve(args.host, args.port, args.allowed_origins))


if __name__ == "__main__":
    main()
