# Demo website

A page that plays a recording and shows its chords as the real-time system does: the decoded row
(fixed-lag Viterbi) above the framewise row (the network alone), so the flicker the decoder removes is
visible. One setting picks the look-ahead (the listening prototype's 0.28 / 0.46 / 0.65 s points); a
checkbox really delays playback to match it, and a panel states the delay and what it is made of. A
checkbox adds the GuitarSet lead-sheet chords as a reference row, and a diagram shows the analysis
stages with the run's settings. See [reports/demo.md](../reports/demo.md).

No framework and no build step: `index.html`, `style.css`, `app.js`, `live.js`.

```bash
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
PYTHONPATH=src python scripts/demo_server.py
```

Then open http://localhost:8000. Space plays and pauses; click or drag on the timeline, or use the seek
bar, to seek.

**Live** (with `scripts/demo_server.py`): the browser decodes the recording, streams it to the server at
playback speed over a WebSocket, and the server runs the live chain (`play_along.Analysis`) on it,
sending back each chord as soon as it is decided. Your own file works the same way: any audio file your
browser can play, up to 100 MB and 10 min. A seek or a new look-ahead starts a new session.

**Precomputed** (any static server, e.g. `python -m http.server -d demo 8000`): no model runs; the page
reveals the chords that `scripts/demo_export.py` computed, each when the live system would have output
it. Own files need the live server.

`demo/samples/` is git-ignored: a copy of each GuitarSet recording (CC BY 4.0) and its JSON
(`chordtracker.demo`: segments and delays of each operating point, framewise segments, the GuitarSet
lead-sheet chords, metadata with the pipeline settings). Only GuitarSet validation takes can be exported.

## Deploy with Docker

The `Dockerfile` at the repo root packages the live server: the page, the prepared samples, the model
(`results/v7_2/c2_la2_seed0`), its configs and the Billboard counts, with CPU-only PyTorch and the
packages of `requirements-demo.txt`. `.dockerignore` lets in only those files, so `data/` and the other
results never enter the build. `demo/samples/` is git-ignored, so export it first (above): the build
stops if `demo/samples/index.json` is missing.

```bash
docker build -t chord-tracker-demo .
docker run --rm -p 8000:8000 chord-tracker-demo        # then open http://localhost:8000
```

The image is built for the machine that builds it (arm64 on Apple silicon). For a linux/amd64 host:

```bash
docker buildx build --platform linux/amd64 -t chord-tracker-demo:amd64 .
```

The server runs as a non-root user and listens on 0.0.0.0 in the container; `GET /healthz` is its health
check. Settings (environment variables, or the same flags of `scripts/demo_server.py`):

| Variable | Default | |
|---|---|---|
| `CHORD_DEMO_HOST` | `127.0.0.1` (`0.0.0.0` in the image) | address to listen on (`--host`) |
| `CHORD_DEMO_PORT`, else `PORT` | `8000` | port (`--port`) |
| `CHORD_DEMO_ALLOWED_ORIGINS` | none | comma-separated origins, besides the server's own, whose pages may open the live route (`--allowed-origins`) |
| `CHORD_DEMO_MAX_SESSIONS` | `8` | live sessions at once |
| `CHORD_DEMO_MAX_SECONDS` | `600` | audio per session, in seconds |
| `CHORD_DEMO_LIVE_MARGIN` | `0.15` | seconds the page adds to its playback delay for the network and the server; raise it for a slow or distant host |

Each listener costs a fraction of one core: per frame of 93 ms of audio, about 6 ms on an Apple silicon
Mac, natively and in the image alike (the image keeps OpenMP and OpenBLAS to one thread; at their defaults
they ran about 6 times slower in Docker Desktop's VM). The block at 10 s into a session, where the tuning
is estimated, takes about 0.1 s (mostly building the tuned CQT kernels). Behind
an HTTPS proxy the page opens the live route as `wss://`; the proxy must pass WebSocket upgrades and
keep the request's `Host`, which the origin check compares with the page's origin.
