"""Demo website with uploads: serves demo/ and analyzes an uploaded recording with the prepared samples' chain.

    python scripts/demo_server.py            # then open http://localhost:8000
    python scripts/demo_server.py --port 8010

The page alone (python -m http.server -d demo) plays the prepared samples; uploading a file needs the
model, so this server adds two routes to the static files:

    GET  /api/health      {"formats": [".wav", ...], "maxBytes": ..., "maxSeconds": ...}
    POST /api/analyze     body: the file's bytes; header X-Filename: its name (URL-encoded)
                          200: the JSON of chordtracker.demo.result (no reference, "audio": null: the
                          browser plays its own copy of the file); 400: unsupported or unreadable file,
                          413: too large; 500: the analysis failed. Errors: {"error": message}

The analysis is demo_export.Analyzer, the same chain and model as the prepared samples, one file at a
time. Formats: those librosa reads here through libsndfile with the same duration and timing as the
original (checked on a GuitarSet take: WAV, FLAC, Ogg Vorbis, MP3; AAC / M4A is not read). Listens on
127.0.0.1 only; uploads are written to a temporary file and deleted after the analysis.
"""
import argparse
import json
import sys
import tempfile
import threading
import traceback
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import soundfile as sf

from chordtracker import paths

sys.path.insert(0, str(Path(__file__).parent))
import demo_export  # noqa: E402

FORMATS = (".wav", ".flac", ".ogg", ".mp3")
MAX_BYTES = 50 * 1024 * 1024
MAX_SECONDS = 600.0
UPLOAD_SOURCE = {"dataset": "uploaded file"}


class Handler(SimpleHTTPRequestHandler):
    analyzer: demo_export.Analyzer
    lock = threading.Lock()
    uploads = 0

    def send_json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/health":
            self.send_json(200, {"formats": list(FORMATS), "maxBytes": MAX_BYTES, "maxSeconds": MAX_SECONDS})
            return
        super().do_GET()

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send_json(404, {"error": "Unknown route."})
            return
        name = Path(unquote(self.headers.get("X-Filename", ""))).name or "upload"
        suffix = Path(name).suffix.lower()
        length = int(self.headers.get("Content-Length") or 0)
        if suffix not in FORMATS:
            self.send_json(400, {"error": f"{name}: unsupported file type. Supported: {', '.join(FORMATS)}."})
            return
        if length > MAX_BYTES:
            self.send_json(413, {"error": f"{name}: larger than {MAX_BYTES // 2 ** 20} MB."})
            return
        if length == 0:
            self.send_json(400, {"error": f"{name}: the file is empty."})
            return
        body = self.rfile.read(length)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"upload{suffix}"
            path.write_bytes(body)
            try:
                duration = sf.info(str(path)).duration
            except Exception:
                self.send_json(400, {"error": f"{name}: could not be read as audio."})
                return
            if not 1.0 <= duration <= MAX_SECONDS:
                self.send_json(400, {"error": f"{name}: {duration:.1f} s long; the demo takes 1 s to "
                                              f"{MAX_SECONDS / 60:.0f} min."})
                return
            try:
                with Handler.lock:                  # one analysis at a time
                    Handler.uploads += 1
                    out = self.analyzer.analyze(path, f"upload-{Handler.uploads}", name, None, source=UPLOAD_SOURCE)
            except Exception:
                traceback.print_exc()
                self.send_json(500, {"error": f"{name}: the analysis failed (see the server's output)."})
                return
        self.send_json(200, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    Handler.analyzer = demo_export.Analyzer()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(paths.REPO_ROOT / "demo")))
    print(f"Demo: http://localhost:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
