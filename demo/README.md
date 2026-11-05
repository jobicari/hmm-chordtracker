# Demo website

A static page that plays a prepared recording and follows its decoded chords (HMM, fixed-lag Viterbi):
the chord now and next, and a timeline with a playhead. Phase 1 of the demo specification; see
[reports/demo.md](../reports/demo.md) for what it shows, how it was checked and what is open.

No framework and no build step: `index.html`, `style.css`, `app.js`. The page computes no chord; every
chord comes from the JSON that `scripts/demo_export.py` writes next to the audio.

```bash
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
python -m http.server -d demo 8000
```

Then open http://localhost:8000. Space plays and pauses; click or drag on the timeline, or use the seek
bar, to seek.

`demo/samples/` is git-ignored: a copy of each GuitarSet recording (CC BY 4.0) and its JSON
(`chordtracker.demo`: decoded and framewise segments, the GuitarSet annotation, metadata). Only GuitarSet
validation takes can be exported.
