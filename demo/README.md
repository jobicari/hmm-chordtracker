# Demo website

A page that plays a recording and shows its chords as the real-time system does: the decoded row
(fixed-lag Viterbi) above the framewise row (the network alone), so the flicker the decoder removes is
visible. One setting picks the look-ahead (0.28 / 0.46 / 0.65 / 1.02 s: the network alone and fixed-lag
Viterbi at lag 2 / 4 / 8; V7.2 has the first three); a checkbox really delays playback to match it, and a
panel states the delay and what it is made of. With the delay on, another checkbox shows each chord 1 s
before you hear it (the lead the listening prototype ended on, reports/play_along.md): the delay grows by
that second, so the chords shown early are already decided. A
checkbox adds the GuitarSet lead-sheet chords as a reference row, and a diagram shows the analysis
stages with the run's settings. See [reports/demo.md](../reports/demo.md).

The model is V14's network with an N class (`results/v14/n_seed0`: 60 chords, major, minor, 7, maj7 and
m7 in 12 keys, plus N.C.), decoded at its best setting on validation (N weight x0.1, chord to N rate
x0.5); N.C. is the network's own call, so it can land on a real chord (reports/v14.md). V7.2's 24-chord
network (`results/v7_2/c2_la2_seed0`, N.C. from a level gate) stays available with `--model v7_2`.
`results/v14/n_seed0/model.pt` is not in git: copy it into that folder before exporting or building.

No framework and no build step: `index.html`, `style.css`, `app.js`, `live.js`.

```bash
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
PYTHONPATH=src python scripts/demo_server.py
```

(`--model v7_2` on both for V7.2: the prepared samples and the server must use the same model.)

Then open http://localhost:8000. Space plays and pauses; click or drag on the timeline, or use the seek
bar, to seek.

**Live** (with `scripts/demo_server.py`): the browser decodes the recording, streams it to the server at
playback speed over a WebSocket, and the server runs the live chain (`play_along.Analysis`) on it,
sending back each chord as soon as it is decided. Your own file works the same way: any audio file your
browser can play, up to 100 MB and 10 min. The page plays the browser's decoding of it, as a WAV, not
the file itself: compressed files (FLAC and MP3 in Safari, MP3 without a seek index elsewhere) seek only
approximately, so after a seek the player would play other samples than the ones the server decodes. A
seek or a new look-ahead starts a new session; a drag on the seek bar or the timeline starts one, on
release.

`tests/test_demo_e2e.py` checks this in Chromium (it needs Playwright: `pip install playwright` and
`playwright install chromium`; skipped without it).

**Precomputed** (any static server, e.g. `python -m http.server -d demo 8000`): no model runs; the page
reveals the chords that `scripts/demo_export.py` computed, each when the live system would have output
it. Own files need the live server.

`demo/samples/` is git-ignored: a copy of each GuitarSet recording (CC BY 4.0) and its JSON
(`chordtracker.demo`: segments and delays of each operating point, framewise segments, the GuitarSet
lead-sheet chords, metadata with the pipeline settings). Only GuitarSet validation takes can be exported.

## Deploy with Docker

The `Dockerfile` at the repo root packages the live server: the page, the prepared samples, the model
(`results/v14/n_seed0`: model.pt, prior.json, run.json, n_counts.json; and `results/v7_2/c2_la2_seed0`), their
configs and the Billboard counts, with CPU-only PyTorch and the packages of `requirements-demo.txt`.
`.dockerignore` lets in only those files, so `data/` and the other results never enter the build.
`demo/samples/` is git-ignored and V14's `model.pt` is not in git, so export the samples first (above) and copy the
weights in: the build stops if `demo/samples/index.json` or `results/v14/n_seed0/model.pt` is missing.

```bash
docker build -t chord-tracker-demo .
docker run --rm -p 8000:8000 chord-tracker-demo        # then open http://localhost:8000
```

The image is built for the machine that builds it (arm64 on Apple silicon). For a linux/amd64 host:

```bash
docker buildx build --platform linux/amd64 -t chord-tracker-demo:amd64 .
```

The server runs as a non-root user and listens on 0.0.0.0 in the container; `GET /healthz` is its health
check (503 while no decoding worker is up). The server process only serves the page and moves bytes; the
decoding runs in worker processes, each loading the model once, and each live session stays on one
worker. A worker decodes one block at a time, so one worker per CPU uses the machine. A worker that
stops is restarted, and its listeners get an error and can start again.

Settings are environment variables (or the same flags of `scripts/demo_server.py`), read when the server
starts: change them with `docker run -e ...` and restart the container, no rebuild needed. The server
logs the values it uses, and `/healthz` reports them (`workers`, `maxSessions`, `liveMarginSeconds`,
`model`, `silenceDbfs`).

| Variable | Default | |
|---|---|---|
| `CHORD_DEMO_HOST` | `127.0.0.1` (`0.0.0.0` in the image) | address to listen on (`--host`) |
| `CHORD_DEMO_PORT`, else `PORT` | `8000` | port (`--port`) |
| `CHORD_DEMO_ALLOWED_ORIGINS` | none | comma-separated origins, besides the server's own, whose pages may open the live route (`--allowed-origins`) |
| `CHORD_DEMO_WORKERS` | the CPUs available, at most 8 | decoding worker processes (`--workers`); any value set is used as given |
| `CHORD_DEMO_MAX_SESSIONS` | `8` | live sessions at once, over all workers |
| `CHORD_DEMO_MAX_SECONDS` | `600` | audio per session, in seconds |
| `CHORD_DEMO_LIVE_MARGIN` | `0.15` | seconds the page adds to its playback delay for the network and the server; raise it for a slow or distant host |
| `CHORD_DEMO_MODEL` | `v14` | `v14`: V14's network with its own N.C.; `v7_2`: V7.2's 24 chords (`--model`). The prepared samples must be exported with the same model |
| `CHORD_DEMO_SILENCE_DBFS` | off for `v14`, `-55` for `v7_2` | the silence gate: a frame is N.C. (no chord) once the input has stayed below this level (dBFS) for 3 frames; `-inf` turns it off (reported as `null`). The 10 s the tuning is estimated on start at the first frame at or above -55 dBFS either way |

```bash
docker run --rm -p 8000:8000 -e CHORD_DEMO_WORKERS=2 -e CHORD_DEMO_MAX_SESSIONS=4 chord-tracker-demo
```

**Sizing.** Each listener costs about 6 ms of CPU per frame of 93 ms of audio on an Apple silicon Mac
(natively and in the image alike: the image keeps OpenMP and OpenBLAS to one thread; at their defaults
they ran about 6 times slower in Docker Desktop's VM). The block 10 s after the first audible frame,
where the tuning is estimated, takes about 0.1 s (mostly building the tuned CQT kernels). Memory: about 50 MB
for the server process and about 370 MB per worker (8 workers: 3 GB; 2 workers: 0.7 GB). So a host
needs one CPU and about 0.4 GB per worker; with 8 workers on that Mac, 8 listeners at once had a worst
lag under 0.1 s.

**If the server is overloaded** (chords late, `scripts/demo_bench.py` exits 1, or memory runs out):
lower `CHORD_DEMO_MAX_SESSIONS` or `CHORD_DEMO_WORKERS`, or raise `CHORD_DEMO_LIVE_MARGIN`, then restart
the container.

**Benchmark a host.** `scripts/demo_bench.py` times the live chain on the prepared samples, one frame at
a time, and with `--sessions N` runs N listeners at once at playback speed through the server's worker
processes. It prints the worst lag (how long after a frame's audio its chords are out), suggests
`CHORD_DEMO_MAX_SESSIONS` and `CHORD_DEMO_LIVE_MARGIN`, and exits 1 if the worst lag is beyond the
server's margin:

```bash
docker run --rm chord-tracker-demo python scripts/demo_bench.py --sessions 8
```

Behind an HTTPS proxy the page opens the live route as `wss://`; the proxy must pass WebSocket upgrades
and keep the request's `Host`, which the origin check compares with the page's origin.
