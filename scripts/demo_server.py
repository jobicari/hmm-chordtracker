"""Demo website with live decoding: serves demo/ and decodes the audio the page streams to it, as it plays.

    python scripts/demo_server.py            # then open http://localhost:8000
    python scripts/demo_server.py --port 8010
    python scripts/demo_server.py --host 0.0.0.0     # reachable from other machines (the Docker image)

The page decodes the recording (a prepared sample or the listener's own file), resamples it to the
model's rate and streams it at playback speed; this server runs the listening prototype's live chain on
it (play_along.Analysis: block-by-block features, the causal CRNN, the three operating points' decoders,
the silence gate if on) and sends every chord back as soon as it is decided. With v15 (the default) and v14
N (no chord) is the network's own class; with v7_2 it comes from the silence gate. Routes:

    GET  /api/health    {"live": true, "points": [...], "metadata": {...}, "maxSeconds": ...,
                        "liveMarginSeconds": ..., "model": ..., "silenceDbfs": ..., "maxSessions": ...,
                        "workers": {"alive": ..., "count": ...}}: what a page needs to decode a file of its
                        own (the same fields as a sample's JSON), the margin it adds to its playback delay,
                        the model, the silence gate's level (null: off), and this server's capacity
    GET  /healthz       the same, for a container's health check: 503 while no worker is up
    WS   /api/live      one session per connection:
                          client {"type": "start"}         -> {"type": "ready", "labels": [...], "sampleRate": ...}
                          client binary: 16-bit mono PCM at sampleRate, any block size
                                                           -> {"type": "labels", "points": {id: {"from": k, "labels": [...]}}}
                          client {"type": "end"}           -> the last labels (decoders flushed), then {"type": "done"}
                          errors                           -> {"type": "error", "message": ...}, then the server closes

This process only serves files and moves bytes (it loads no model). The decoding runs in worker processes
(chordtracker.live_workers), each loading the chain once: a session is pinned to the least-loaded worker
for its lifetime and has its own features, network state and decoders there. A worker runs one block at
a time, so a worker per CPU uses the machine; a worker that dies ends its sessions with an error and is
restarted. At most max sessions run at once (over all workers) and each stops after max seconds of
audio. Only the page's own files and samples are served.

Settings, each a flag or an environment variable (read at start, so a container takes new values on
restart, without a rebuild):

    --host              CHORD_DEMO_HOST              127.0.0.1 (this machine only)
    --port              CHORD_DEMO_PORT, else PORT   8000
    --allowed-origins   CHORD_DEMO_ALLOWED_ORIGINS   none: comma-separated origins (e.g. "https://example.github.io")
                                                     whose pages may use the live route, besides the server's own
    --workers           CHORD_DEMO_WORKERS           the CPUs available, at most 8
    --max-sessions      CHORD_DEMO_MAX_SESSIONS      8
    --max-seconds       CHORD_DEMO_MAX_SECONDS       600: audio per session
    --live-margin       CHORD_DEMO_LIVE_MARGIN       0.15: s the page adds to its playback delay for the network
                                                     and the server (more for a slow or distant host)
    --model             CHORD_DEMO_MODEL             v15: V14 fine-tuned for dim, aug and hdim7 (97 classes,
                                                     results/v15/ft_seed0); v14: V14's network with an N class
                                                     (61 classes, results/v14/n_seed0); v7_2: V7.2's 24-class
                                                     network (results/v7_2/c2_la2_seed0)
    --silence-dbfs      CHORD_DEMO_SILENCE_DBFS      the silence gate's level: N where the input stays below it for 3
                                                     frames; -inf turns it off. Default: off for v15 and v14 (own N),
                                                     -55 (play_along.SILENCE_DBFS) for v7_2
"""
import argparse
import asyncio
import functools
import itertools
import json
import math
import os
import signal
import sys
from pathlib import Path
from urllib.parse import urlsplit

import tornado.web
import tornado.websocket

sys.path.insert(0, str(Path(__file__).parent))
from chordtracker import paths  # noqa: E402
from chordtracker.live_workers import Workers  # noqa: E402

PAGE_FILES = r"/(|index\.html|style\.css|app\.js|live\.js|samples/[^/]+)"     # not README.md
DEFAULT_WORKERS_CAP = 8
MODELS = ("v15", "v14", "v7_2")           # demo.MODELS, and demo_export.DEFAULT_SILENCE_DBFS below (not imported:
DEFAULT_MODEL = "v15"              # this process loads no model code)
DEFAULT_SILENCE_DBFS = {"v15": float("-inf"), "v14": float("-inf"), "v7_2": -55.0}


def origin_key(origin: str) -> str:
    """"https://Example.org/" -> "https://example.org": an origin as browsers send it, for comparison."""
    parts = urlsplit(origin.strip())
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def available_cpus() -> int:
    """CPUs this process may use: its CPU affinity, within the container's quota (cgroup v2 cpu.max) if set."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        limit, period = quota.read_text().split()
        if limit != "max":
            cpus = min(cpus, max(1, int(limit) // int(period)))
    return cpus


class Chain:
    """A model's chain (demo_export.Analyzer), loaded once in a worker and warmed up; new_session() -> a
    fresh play_along.Analysis with the silence gate at silence_dbfs (-inf: off; None: the model's default).
    N is the last label (97 labels for v15, 61 for v14, 25 for v7_2)."""

    def __init__(self, model: str = DEFAULT_MODEL, silence_dbfs: float | None = None):
        import torch
        import demo_export
        from chordtracker import demo
        torch.set_num_threads(1)                               # one block at a time per worker
        self.demo = demo
        self.silence_dbfs = DEFAULT_SILENCE_DBFS[model] if silence_dbfs is None else silence_dbfs
        self.analyzer = a = demo_export.Analyzer(model)
        self.labels, self.points = a.labels, a.points
        self.sr, self.hop = a.sr, a.hop
        self.metadata = {"sampleRate": self.sr, "hopLength": self.hop, "featureName": demo_export.FEATURE,
                         "model": str(a.run.relative_to(paths.REPO_ROOT)),
                         "inputLookaheadFrames": demo_export.INPUT_LOOKAHEAD,
                         "frameSeconds": round(self.hop / self.sr, 6),
                         "pipeline": demo_export.pipeline_settings(a, self.silence_dbfs)}
        self.warm_up()

    def warm_up(self) -> None:
        """One throwaway session past the tuning step (quiet noise), so that the first listener does not
        pay for the first calls (numba, torch, the tuned kernels): 0.2 s on the first frame otherwise."""
        import numpy as np
        n = int(self.analyzer.cfg["cqt"]["tuning_seconds"] * self.sr) + 4 * self.hop
        session = self.new_session()
        session.step((0.01 * np.random.default_rng(0).standard_normal(n)).astype(np.float32))
        session.finish()

    def new_session(self):
        return self.analyzer.session(self.silence_dbfs)

    def info(self) -> dict:
        """What the server needs from the chain: the page's fields, the class labels, the sample rate."""
        return {"points": self.demo.point_info(self.sr, self.hop, self.points), "defaultPoint": self.demo.DEFAULT_POINT,
                "framewiseDecisionDelaySeconds": self.demo.framewise_decision_delay(self.sr, self.hop),
                "metadata": self.metadata, "labels": self.labels, "sampleRate": self.sr}


def health(workers: Workers, settings) -> dict:
    info = workers.info or {}
    return {"live": True, **{k: info.get(k) for k in ("points", "defaultPoint", "framewiseDecisionDelaySeconds",
                                                       "metadata")},
            "maxSeconds": settings.max_seconds, "liveMarginSeconds": settings.live_margin,
            "model": settings.model,
            "silenceDbfs": settings.silence_dbfs if math.isfinite(settings.silence_dbfs) else None,
            "maxSessions": settings.max_sessions, "workers": {"alive": workers.alive, "count": workers.n}}


class PageFileHandler(tornado.web.StaticFileHandler):
    """The page's files and the samples, revalidated on every load ("no-cache", answered 304 from the ETag
    when unchanged): without it browsers keep using a cached app.js after a new image is deployed.

    The ETag is the file's modification time and size, read on every request: tornado's own is a hash it
    computes once per process, so a file edited while the server runs kept its old ETag and got 304."""

    def set_extra_headers(self, path):
        self.set_header("Cache-Control", "no-cache")

    def compute_etag(self):
        st = os.stat(self.absolute_path)
        return f'"{st.st_mtime_ns:x}-{st.st_size:x}"'


class HealthHandler(tornado.web.RequestHandler):
    def initialize(self, workers, settings, strict=False):
        self.workers, self.settings_, self.strict = workers, settings, strict

    def get(self):
        self.set_header("Cache-Control", "no-store")
        if self.strict and not self.workers.alive:
            self.set_status(503)
        self.write(health(self.workers, self.settings_))


class LiveHandler(tornado.websocket.WebSocketHandler):
    """One live session: PCM blocks in, decided chords out (decoded by the worker the session is pinned to)."""
    sessions = 0
    ids = itertools.count()

    def initialize(self, workers, settings, allowed_origins):
        self.workers, self.settings_, self.allowed_origins = workers, settings, allowed_origins
        self.sid, self.samples, self.counted = None, 0, False

    def check_origin(self, origin):
        """The server's own page (tornado's same-origin check), or a page from an allowed origin."""
        return origin_key(origin) in self.allowed_origins or super().check_origin(origin)

    def on_message(self, message):
        if isinstance(message, bytes):
            self.on_audio(message)
        else:
            self.on_command(json.loads(message))

    def on_command(self, msg: dict):
        kind = msg.get("type")
        if kind == "start":
            if self.sid is not None:
                return self.fail("The session has already started.")
            if LiveHandler.sessions >= self.settings_.max_sessions:
                return self.fail("The demo server is busy; try again in a moment.")
            sid = next(LiveHandler.ids)
            if not self.workers.open(sid, self.on_worker):
                return self.fail("The decoder is starting; try again in a moment.")
            self.sid, self.counted = sid, True
            LiveHandler.sessions += 1
            info = self.workers.info
            self.send({"type": "ready", "labels": info["labels"], "sampleRate": info["sampleRate"]})
        elif kind == "end":
            if self.sid is None:
                return self.fail("No session.")
            self.workers.end(self.sid)
        else:
            self.fail(f"Unknown message {kind!r}.")

    def on_audio(self, data: bytes):
        if self.sid is None:
            return self.fail("Send {\"type\": \"start\"} first.")
        self.samples += len(data) // 2
        if self.samples > self.settings_.max_seconds * self.workers.info["sampleRate"]:
            return self.fail(f"The demo decodes at most {self.settings_.max_seconds / 60:.0f} min per session.")
        self.workers.audio(self.sid, data)

    def on_worker(self, kind: str, payload):
        """A message from the session's worker: new labels, the end of the session, or an error."""
        if kind == "labels" and payload:
            self.send({"type": "labels", "points": payload})
        elif kind == "done":
            self.send({"type": "done"})
        elif kind == "error":
            self.fail(payload)

    def send(self, message: dict):
        try:
            self.write_message(message)
        except tornado.websocket.WebSocketClosedError:
            pass

    def fail(self, message: str):
        self.send({"type": "error", "message": message})
        self.close()

    def on_close(self):
        if self.sid is not None:
            self.workers.close(self.sid)
        if self.counted:
            LiveHandler.sessions -= 1
            self.counted = False


def make_app(workers: Workers, static_dir: Path, settings) -> tornado.web.Application:
    shared = {"workers": workers, "settings": settings}
    origins = {origin_key(o) for o in settings.allowed_origins}
    return tornado.web.Application([
        (r"/api/health", HealthHandler, shared),
        (r"/healthz", HealthHandler, {**shared, "strict": True}),
        (r"/api/live", LiveHandler, {**shared, "allowed_origins": origins}),
        (PAGE_FILES, PageFileHandler, {"path": str(static_dir), "default_filename": "index.html"}),
    ], websocket_max_message_size=2 ** 20)


def parse_args(argv=None, env=os.environ) -> argparse.Namespace:
    """The command line, with defaults from the environment (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=env.get("CHORD_DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(env.get("CHORD_DEMO_PORT", env.get("PORT", 8000))))
    parser.add_argument("--allowed-origins", default=env.get("CHORD_DEMO_ALLOWED_ORIGINS", ""),
                        help="comma-separated origins whose pages may use the live route")
    parser.add_argument("--workers", type=int,
                        default=int(env.get("CHORD_DEMO_WORKERS", min(available_cpus(), DEFAULT_WORKERS_CAP))))
    parser.add_argument("--max-sessions", type=int, default=int(env.get("CHORD_DEMO_MAX_SESSIONS", 8)))
    parser.add_argument("--max-seconds", type=float, default=float(env.get("CHORD_DEMO_MAX_SECONDS", 600.0)))
    parser.add_argument("--live-margin", type=float, default=float(env.get("CHORD_DEMO_LIVE_MARGIN", 0.15)))
    parser.add_argument("--model", choices=MODELS, default=env.get("CHORD_DEMO_MODEL", DEFAULT_MODEL))
    parser.add_argument("--silence-dbfs", type=float,
                        default=float(env["CHORD_DEMO_SILENCE_DBFS"]) if "CHORD_DEMO_SILENCE_DBFS" in env else None)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_sessions < 1:
        parser.error("--workers and --max-sessions must be at least 1")
    if args.model not in MODELS:
        parser.error(f"CHORD_DEMO_MODEL must be one of {', '.join(MODELS)}")
    if args.silence_dbfs is None:
        args.silence_dbfs = DEFAULT_SILENCE_DBFS[args.model]
    args.allowed_origins = [o for o in (s.strip() for s in args.allowed_origins.split(",")) if o]
    return args


async def serve(settings) -> None:
    workers = Workers(functools.partial(Chain, settings.model, settings.silence_dbfs), settings.workers)
    await workers.start()
    server = make_app(workers, paths.REPO_ROOT / "demo", settings).listen(settings.port, address=settings.host)
    print(f"Demo: http://localhost:{settings.port} (listening on {settings.host}; {settings.workers} decoding "
          f"workers, at most {settings.max_sessions} sessions, live margin {settings.live_margin:g} s, "
          f"model {settings.model}, silence gate "
          f"{'off' if not math.isfinite(settings.silence_dbfs) else f'{settings.silence_dbfs:g} dBFS'})", flush=True)
    stop = asyncio.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    await stop.wait()
    server.stop()
    await workers.stop()


def main() -> None:
    asyncio.run(serve(parse_args()))


if __name__ == "__main__":
    main()
