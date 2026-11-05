# Demo website

## Question

The listening prototype (reports/play_along.md) shows the chords live in a desktop window and needs a
virtual audio cable. **A web page that plays a prepared recording and shows the decoded chords in time
with it** lets a reviewer who has never seen the code see what the system does. This report covers
Phase 0 (inspection) and Phase 1 (stable prepared-sample flow) of the demo specification; Phases 2 to 4
(framewise comparison row, pipeline and reference rows, upload) are not started.

## Phase 0: what the repo had

| Spec question | Finding |
|---|---|
| Frontend framework and entry point | None. The repo is Python only. The only interface is `scripts/play_along.py`, a tkinter window. `ableton_device/` is an empty, untracked folder. |
| Audio playback | Only in the prototype: a sounddevice output stream, delayed by the total look-ahead plus a lead. Nothing for a browser. |
| Where framewise outputs are | Not stored. Each run script recomputes the network's frame posteriors in memory (`train_v7_2.song_posteriors`, `run_v7.frame_paths`); the prototype computes them block by block (`play_along.Analysis`). `results/<version>/<split>/` keeps only per-song scores (`per_song.csv`) and summaries. |
| Where HMM/Viterbi outputs are | Not stored either: `online_hmm.fixed_lag_viterbi` (whole song) or `online_hmm.FixedLagViterbi` (live) decode them in memory. The model and its training prior are in `results/v7_2/c2_la2_seed0/`. |
| Timing | One label per frame of 2,048 samples at 22,050 Hz (92.9 ms); frame t sits at t x hop / sr (`acoustic_hmm.frame_times`). Runs of frames become segments with boundaries halfway between frames (`evaluate.path_to_annotation`), which is also how the WCSR is computed. Both outputs share that frame grid. Labels are aligned: frame t's label is known its total look-ahead later (0.28 s for the network alone, 0.65 s for the fixed-lag Viterbi at lag 4), so a live display needs playback delayed by that much (reports/v7_1.md, "The target set-up"). |
| Prepared audio and verified annotations | GuitarSet (CC BY 4.0, `data/guitarset/`, git-ignored): mono mic recordings and their annotations, the ones every version is scored on, so recording and annotation are known to match. The Isophonics Beatles annotations (`data/beatles/`) have no audio in the repo (the prototype takes the music from Spotify) and RWC-P audio is not redistributable, so neither can back the demo. |
| Cause of the flickering | No page existed, so no rendering flicker to trace. The flicker the project knows is in the output: the network alone changes chord about twice as often as the music (64 vs 22 changes per minute on the Funk take below), in runs of one or two frames; the fixed-lag Viterbi halves that. A page would add its own flicker if it redrew the chord or the timeline on every playback update. |
| Smallest Phase 1 plan | Export the outputs of a few GuitarSet validation takes with the real chain to JSON next to the audio, in the spec's result shape; a static page (no framework, no build) that plays the audio, draws the decoded timeline once, and on each playback update moves only the playhead and changes the chord text when the active segment changes. |

The spec's header suggests "Chroma features · Chord templates · HMM · Viterbi". That describes V0 to V4,
not the current system: the page says "Log-CQT features · CRNN chord classifier · HMM · Fixed-lag
Viterbi". The spec's description calls the system "interpretable"; the page leaves that word out, since
the acoustic model is a neural network.

## Phase 1: what was built

    scripts/demo_export.py      runs the prototype's chain on whole validation takes, writes demo/samples/
    src/chordtracker/demo.py    frame paths -> segments, the result JSON (tests/test_demo.py)
    demo/index.html, app.js, style.css, README.md

The chain is the listening prototype's: `streaming_features.offline_log_cqt` (the columns
`StreamingFeatures` gives block by block), the V7.2 network with input look-ahead 2 on the C2 grid
(seed 0), then the argmax per frame (framewise) and the fixed-lag Viterbi with lag 4 and p_self 0.98
(decoded), the settings of `play_along.OPERATING_POINTS`. Each sample's JSON follows the spec's
TrackerResult: `duration`, `decoded`, `framewise`, `reference`, `metadata` (sample rate, hop, frame
count, model and decoder read from the run). Segments carry the label as output (`chord`, a Harte label
such as `G#:min`) and as shown (`name`, `G#m`). Framewise runs are merged into segments but never
smoothed. The JSON already holds the framewise and reference segments; Phase 1 shows only the decoded
ones.

The page:

- **Clock**: `audio.currentTime` is the only time source; playhead, seek bar, time display and decoded
  chord are derived from it on every animation frame while playing, and at once on pause, seek and load.
- **Active segment**: start <= t < end, so at a boundary the new chord is active; at or past the end
  the last segment stays active.
- **No flicker**: the timeline is built once per sample; during playback only the playhead's transform,
  the seek bar and changed texts are written, and the chord text and active highlight only when the
  active segment changes.
- **Long recordings**: at least 40 px per second, scrolling horizontally; the view follows the playhead
  only when it leaves the visible part.
- **Stale results**: picking a sample clears the old one before loading; a newer pick makes an older,
  slower load drop its results; a failed load leaves an error and an empty page.
- **Seeking**: the audio is fetched whole and played from a blob URL. `python -m http.server` does not
  answer byte-range requests, and Chrome then cannot seek a plain URL (seekable range 0 - 0, found while
  testing); a blob is seekable on any static server.

## What was checked

`tests/test_demo.py`: segment boundaries halfway between frames, segments cover the whole recording
without gaps, single-frame framewise runs are kept, metadata from the run, decoded and framewise of
different lengths refused. Full suite in `hmm-chordtracker`: all passing.

In the browser (Funk, Rock and SS takes), against the JSON:

| Check | Result |
|---|---|
| Decoded chord, active highlight (exactly one) and playhead position vs the JSON at the audio time, every ~16 ms of playback (0 - 5 s, 9 - 12 s, 0 - 3 s, then pause / resume 9 - 13.5 s) | 0 mismatches; playhead never moved back |
| Timeline rebuilt during playback | never (0 child-list mutations on the track) |
| Chord text rewrites | only at segment changes and seeks |
| Click on the timeline at 12 s, seek bar to 20.1 s, Restart | audio, playhead, seek bar, time and chord all agree |
| Playback to the end; play again | ends at 25.9 / 25.9 s with the last chord, status "Ended"; plays again from 0 |
| Three sample picks 5 ms apart | the last pick wins, its 16 segments shown, no mix |
| Missing sample file | error in the status line, nothing of the previous sample left |
| Console | no page errors; the browser pane itself requests `/favicon.ico` (404) although the page declares no icon file |

## Open questions

- **Playback delay.** The page plays the recording undelayed and shows the aligned labels, as if
  playback were delayed by the look-ahead. Whether to say so on the page, or simulate the delay, is
  open.
- **Reference annotations (Phase 3).** GuitarSet's annotations are the performed chords (sevenths,
  half-diminished...), which the 24-chord system cannot output. Show them as annotated, or reduced to
  major/minor as the WCSR judges them?
- **Which prepared samples.** Three validation takes, chosen by style, not by score; which one to make
  the default is open.
- **Upload (Phase 4)** needs a backend: the analysis is Python (PyTorch, librosa), so a small local
  server would run `demo_export`'s chain on the uploaded file. Formats would be what librosa reads.
- **Narrow segments** shorter than their name hide the text (the name is in the tooltip and in the large
  chord display when active); the spec asks for text labels always.

## Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_demo.py
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
python -m http.server -d demo 8000
```
