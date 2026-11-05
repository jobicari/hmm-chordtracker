# Demo website

A static page that plays a prepared recording and shows its chords as the real-time system would: each
chord appears only once the live decoder would have output it. The decoded row (fixed-lag Viterbi) sits
above the framewise row (the network alone), so the flicker the decoder removes is visible. One setting
picks the look-ahead (the listening prototype's 0.28 / 0.46 / 0.65 s points); a checkbox really delays
playback to match it (the audio waits after Play, Restart or a seek while the system listens ahead),
and a panel states the delay and what it is made of. See [reports/demo.md](../reports/demo.md).

No framework and no build step: `index.html`, `style.css`, `app.js`. The model does not run in the
browser: every chord comes from the JSON that `scripts/demo_export.py` writes next to the audio, and the
page only decides when to reveal it.

```bash
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
python -m http.server -d demo 8000
```

Then open http://localhost:8000. Space plays and pauses; click or drag on the timeline, or use the seek
bar, to seek.

`demo/samples/` is git-ignored: a copy of each GuitarSet recording (CC BY 4.0) and its JSON
(`chordtracker.demo`: segments and decision delay of each operating point, framewise segments, the
GuitarSet annotation, metadata). Only GuitarSet validation takes can be exported.
