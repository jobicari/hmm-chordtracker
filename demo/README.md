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
