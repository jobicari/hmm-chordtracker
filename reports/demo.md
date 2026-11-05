# Demo website

## Question

The listening prototype (reports/play_along.md) shows the chords live in a desktop window and needs a
virtual audio cable. **A web page that plays a prepared recording and shows the decoded chords in time
with it** lets a reviewer who has never seen the code see what the system does. This report covers
Phase 0 (inspection) and Phase 1 (stable prepared-sample flow) of the demo specification, reworked after
review into a real-time view: the chords appear as the live system would output them, with the framewise
row (the spec's Phase 2) brought forward to show what the decoder smooths. The pipeline and reference
rows (Phase 3) and upload (Phase 4) are not started.

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

After the first review the page changed in three ways: the header states the comparison (the network
alone flickers, the Viterbi decoder smooths) instead of listing methods; the page is real time rather
than a finished track; and one setting picks the look-ahead.

## Phase 1: what was built

    scripts/demo_export.py      runs the prototype's chain on whole validation takes, writes demo/samples/
    src/chordtracker/demo.py    frame paths -> segments, each point's delays, the result JSON (tests/test_demo.py)
    demo/index.html, app.js, style.css, README.md

**Outputs.** The chain is the listening prototype's: `streaming_features.offline_log_cqt` (the columns
`StreamingFeatures` gives block by block), the V7.2 network with input look-ahead 2 on the C2 grid
(seed 0), then the prototype's three operating points (`play_along.OPERATING_POINTS`): the network
alone (0.28 s of look-ahead), fixed-lag Viterbi lag 2 with p_self 0.9 (0.46 s) and lag 4 with p_self
0.98 (0.65 s, the default, the steadiest by ear). `online_hmm.fixed_lag_viterbi` makes the same
decisions as the streaming decoder (`tests/test_online_hmm.py`). Each sample's JSON holds, per point,
the segments and the decision delay (total look-ahead + one frame, `play_along.required_delay`), and the
framewise segments (argmax per frame, runs merged, never smoothed).

**Real time.** The model does not run in the browser; the page says so and reveals each label only when
the live system would have output it. A fixed-lag decision is final, so this is exactly the live output.
With h one frame (92.9 ms) and D a row's decision delay, frame f is decided once the system has heard
f h + D. Two clocks, both from `audio.currentTime` (s):

    heard        delay on: after Play, Restart or a seek the audio really waits P = D + h / 2 (0.79 s
                 at lag 4) while the input runs alone, then heard = s + P, also across pause and
                 resume; delay off: heard = s. The end of the recording once reached (the decoder flushes)
    shown(row)   up to min(s, end of the last decided frame); a striped mask covers the rest

The extra half frame in P is there because frame f is drawn from (f - 1/2) h, halfway from the previous
frame, but decided at f h + D: with D alone each new chord would appear up to 46 ms after it starts being
heard; with P every chord is on screen when it is heard. With the delay off, the decoded chord comes D
late (0.74 s at lag 4) and the framewise one 0.37 s late. A dashed line marks what the system has heard,
a red one what you hear. A panel under the settings states the delay applied, that the audio is held
back by it, and what it is made of (network look-ahead + decoder lag + one frame + half a frame), all
read from the JSON.

**Page.** Header with the comparison; a notice that the chords were computed ahead of time; sample,
look-ahead (three buttons) and the delay checkbox, with a sentence that states the delay or its absence;
the chord now (decoded) and the network's chord now; play, restart, seek bar, time; the decoded row above
the framewise row on one time axis, click or drag to seek. No accuracy figure is shown. Tracks are built
once per sample (the decoded row again when the look-ahead changes); during playback only transforms,
the seek bar and texts that change are written. Audio is played from a blob URL, so it can be seeked on
`python -m http.server` (no byte ranges there: Chrome reports a seekable range of 0 - 0 for a plain URL).

## What was checked

`tests/test_demo.py`: segment boundaries halfway between frames, segments cover the recording without
gaps, single-frame framewise runs kept, look-ahead and decision delay of each point, outputs of
different lengths or missing points refused. Full suite in `hmm-chordtracker`: all passing.

In the browser (Funk, Rock and SS takes):

| Check | Result |
|---|---|
| Reveal rule over the Funk take, every 13.7 ms, 3 points x delay on / off x 2 rows (31,140 checks, with the delay at D): no frame shown before f h + D, and none withheld once decided and reached | 0 early, 0 withheld |
| Delay on, P = D + h / 2: Play from 0 | audio started 792 ms after Play (P = 789 ms); then 0 ms between what you hear and what is shown, 0 early, over 2.5 s |
| Pause and resume; seek while playing; look-ahead change while playing; pause during the wait | resume plays at once; a seek and a look-ahead change wait again (the new P, 0.42 s at 0.28 s); a pause during the wait stops it |
| During playback (3 points x delay on / off): chord texts, single active block, mask positions vs the JSON, every ~16 ms | 0 mismatches |
| Blocks rebuilt during playback | never |
| Playing to the end with the delay off | the decoder flushes: the whole row shown, last chord, status "Ended" |
| Three sample picks 5 ms apart | the last pick wins; nothing of the others shown; look-ahead and delay settings kept |
| Console | no page errors; the browser pane itself requests `/favicon.ico` (404) although the page declares no icon file |

## Open questions

- **Half a frame.** The delay includes half a frame so that chords are on time with segments cut halfway
  between frames, as the WCSR cuts them. Whether the causal features' frame time convention would put
  the frame elsewhere was not checked; the prototype's delay (`play_along.required_delay`) has no half
  frame.
- **Seeking is not a restart.** After a seek the page shows the decisions of the continuous run; a live
  system started at that point would begin with an empty decoder and running level.
- **After a seek while paused**, the input restarts, so the last D seconds before the playhead are masked
  until Play: the live system has not decided them yet.
- **Reference annotations (Phase 3).** GuitarSet's chords as played (sevenths, half-diminished...) or
  reduced to major/minor as the WCSR judges them?
- **Upload (Phase 4)** needs a small local server that runs `demo_export`'s chain on the file.
- **Narrow blocks** hide their name (tooltip, and the large chord when it is the current one); the spec
  asks for text labels always.

## Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_demo.py
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
python -m http.server -d demo 8000
```
