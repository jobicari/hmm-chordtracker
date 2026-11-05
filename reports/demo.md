# Demo website

## Question

The listening prototype (reports/play_along.md) shows the chords live in a desktop window and needs a
virtual audio cable. **A web page that plays a prepared recording and shows the decoded chords in time
with it** lets a reviewer who has never seen the code see what the system does. This report covers
Phase 0 (inspection) and Phase 1 (stable prepared-sample flow) of the demo specification, reworked after
review into a real-time view: the chords appear as the live system would output them, with the framewise
row (the spec's Phase 2) brought forward to show what the decoder smooths. Phase 3 (pipeline section and
reference row) and Phase 4 (upload, errors, acceptance checks) are done.

## Phase 0: what the repo had

| Spec question | Finding |
|---|---|
| Frontend framework and entry point | None. The repo is Python only. The only interface is `scripts/play_along.py`, a tkinter window. `ableton_device/` is an empty, untracked folder. |
| Audio playback | Only in the prototype: a sounddevice output stream, delayed by the total look-ahead plus a lead. Nothing for a browser. |
| Where framewise outputs are | Not stored. Each run script recomputes the network's frame posteriors in memory (`train_v7_2.song_posteriors`, `run_v7.frame_paths`); the prototype computes them block by block (`play_along.Analysis`). `results/<version>/<split>/` keeps only per-song scores (`per_song.csv`) and summaries. |
| Where HMM/Viterbi outputs are | Not stored either: `online_hmm.fixed_lag_viterbi` (whole song) or `online_hmm.FixedLagViterbi` (live) decode them in memory. The model and its training prior are in `results/v7_2/c2_la2_seed0/`. |
| Timing | One label per frame of 2,048 samples at 22,050 Hz (92.9 ms); frame t sits at t x hop / sr (`acoustic_hmm.frame_times`). Runs of frames become segments with boundaries halfway between frames (`evaluate.path_to_annotation`), which is also how the WCSR is computed. Both outputs share that frame grid. Labels are aligned: frame t's label is known its total look-ahead later (0.28 s for the network alone, 0.65 s for the fixed-lag Viterbi at lag 4), so a live display needs playback delayed by that much (reports/v7_1.md, "The target set-up"). |
| Prepared audio and verified annotations | GuitarSet (CC BY 4.0, `data/guitarset/`, git-ignored): mono mic recordings and their lead-sheet chord annotations, the ones every version is scored on, so recording and annotation are known to match. The Isophonics Beatles annotations (`data/beatles/`) have no audio in the repo (the prototype takes the music from Spotify) and RWC-P audio is not redistributable, so neither can back the demo. |
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

## Phase 3: pipeline and reference

**Pipeline.** An expandable section lists the stages in the order the code runs them: audio input,
harmonic part from past audio (`features_v7_1.causal_harmonic`), log constant-Q spectrum computed
column by column (`streaming_features`), the CRNN (`model_v6`, causal GRU), the framewise output, the
HMM (evidence = posterior / training prior, `online_hmm.evidence`; transitions from the Billboard
chord-change counts, `transitions`), the fixed-lag Viterbi, and the segments and their display. Every
number in it comes from the sample's `metadata.pipeline`, which `demo_export.pipeline_settings` reads
from the run's config, the code's constants and the counts file; the Viterbi stage follows the chosen
look-ahead (at 0.28 s it says the decoder is not used).

**Reference.** The `.lab` files behind every score are GuitarSet's *lead-sheet* chords (the first chord
annotation of each JAMS file, `GuitarSetAdapter(labels="instructed")`), not the performed ones; checked
on the Funk take (labels and times equal to JAMS annotation 0, different from annotation 1). The first
version of this report called them performed chords; that was wrong and is corrected here and on the
page. They are aligned to the recording (the same timing every WCSR uses). A "Show reference
annotations" checkbox (off by default) adds a row under the two prediction rows: white blocks with dark
text and a coloured base line, so they do not look like model output; labelled "reference labels, not
model output", with the source under the timeline; chords outside the 24 major / minor chords (7, hdim7,
min7...: 6 of 14 on the Funk take) striped, with their actual label. The reference is not masked (it is
not decided by the system); its active block follows what you hear. A sample without annotations gets a
disabled checkbox and "Reference annotations are not available for this recording."

## Phase 4: upload, errors and acceptance

**Upload.** Analysing a file needs the model, so `scripts/demo_server.py` serves `demo/` and adds
`POST /api/analyze` (the file's bytes, its name in a header) and `GET /api/health` (formats and limits).
The analysis is `demo_export.Analyzer`, the same chain and model as the prepared samples, one file at a
time, on 127.0.0.1 only; the upload is deleted afterwards. Uploading the Rock take's WAV gives exactly
the prepared sample's segments for all three points and the framewise row. The page offers a file
button and a drop zone, shows the file name, an "Analyzing name... 3 s" status with a spinner, and
plays the browser's own copy of the file; the result has no reference ("not available"). Served by
`python -m http.server` alone, the upload is greyed out with a note on how to start the server.

**Formats.** Only what was tested is offered: WAV, FLAC, Ogg Vorbis and MP3. Converted from a GuitarSet
take with ffmpeg, each loads here with the original's duration (25.941 s) and no time offset
(cross-correlation with the WAV: 0 samples), and Chrome reports the same duration for the MP3 and Ogg.
AAC / M4A cannot be read by this environment's libsndfile, so it is refused. Limits: 100 MB, 1 s to
10 min. The lossy files give slightly different chords (Ogg: 17 instead of 16 segments at 0.46 s).

**Errors and stale results.** Wrong type or size is refused in the browser before upload; unreadable
audio, a too short or too long file and a failed analysis come back from the server as a message; the
request is stopped after 300 s (a 9.5 min, 100 MB stereo WAV took 103 s to analyse). Every new choice (a sample or a file) clears the page first, aborts a
running analysis and makes any older load drop its result; the uploaded file stays in the list as
"Your file: name" and comes back without a new analysis.

**Notes and limitations.** A short section at the end of the page: major / minor triads only, always a
chord; trained on the RWC-P and GuitarSet training songs, the samples are validation takes; chords are
precomputed and revealed as live, and a seek is not a restart; uploads are analysed locally, lossy
formats change the chords slightly, no reference for them; no scores on the page.

## After review: live decoding on the server

The upload analysis of Phase 4 computed a whole file at once (79 s of features and 3.9 GB of memory for
9.5 min of audio). After review the server decodes live instead, as the system was meant to run, and
the upload route is gone.

**How.** The page decodes the recording with the browser's own decoder, mixes it to mono, resamples it
to 22,050 Hz (an `OfflineAudioContext`) and streams it as 16-bit PCM over a WebSocket (`/api/live`), at
playback speed: never past what the system has heard. `scripts/demo_server.py` (Tornado, already in the
environment as a Jupyter dependency) gives each connection its own `play_along.Analysis` (block-by-block
features, the causal CRNN, the three points' decoders, the model shared) and sends every label as soon
as it is decided; at the end of the recording the page sends "end" and the server flushes the decoders
(`Analysis.finish`, new). The page draws each label as it arrives, only the last block changing; the
precomputed JSON is still used for the reference and the pipeline settings, and as the fallback when no
server answers (the notice says which mode is on). A seek, Restart, a new look-ahead or the delay
checkbox start a new session from what you hear: a restarted live system knows nothing of the past, so
the rows start empty there. This removes the "a seek is not a restart" caveat of the precomputed mode.

**Same chords.** Streamed in blocks and finished, `Analysis` gives exactly the whole-file labels at every
point (`tests/test_play_along.py`, on synthetic audio with a seeded model). Through the server, the Rock
take streamed from Python as 16-bit PCM gives the prepared sample's 16 decoded segments, chord for chord;
in the browser (its own decoder and resampler) the first segments of the Funk take match the JSON's.

**Cost.** One session on the M1 Pro, one thread: the 26 s Rock take as fast as it can be sent in 2.7 s,
about 10 ms per 93 ms frame; memory stays small whatever the length. MAX_SESSIONS = 8 at once, 10 min per
session.

**Delay.** Live, the playback delay is the point's (0.79 s at lag 4) plus a 0.15 s margin for the network
and the server: 0.94 s. Measured locally, each chord reached the page 0.63 to 0.67 s after its moment of
audio, a frame sooner than the decision delay of the precomputed mode (0.74 s): the page streams every
20 ms, so the server does not wait for a whole 2,048-sample block as the prototype's input did. The panel
shows this measurement for every session and says so if a chord came later than the delay allows.

**Own files.** Anything the browser can play (Chrome: WAV, MP3, FLAC, Ogg, M4A / AAC, which the server's
libsndfile could not read), up to 100 MB and 10 min. A file the browser cannot decode gets "this browser
cannot decode it as audio".

## Acceptance checks (spec section 11)

| Criterion | Result | Evidence |
|---|---|---|
| Page opens in a usable initial state | pass | the default sample loads, controls enable on "Ready" |
| Prepared sample loads without an upload | pass | the first sample (or the one in the URL) loads on open |
| Audio and outputs matched | pass | the page refuses rows that do not cover the recording; audio duration = JSON duration (35.548 s) |
| Decoded timeline spans the recording | pass | every row starts at 0 and ends at the audio's end |
| Play / pause | pass | below |
| Current time and duration correct | pass | "0:35.5 / 0:35.5" at the end, from `audio.currentTime` and the JSON |
| Playhead follows playback time | pass | playhead and masks vs `audio.currentTime` every ~16 ms: 0 mismatches |
| Chord changes at the decoded boundaries | pass | 0 ms between what you hear and what is shown (within 1 ms at the end, the JSON's times are rounded to the ms) |
| Seeking consistent | pass | click, drag and seek bar: audio, playhead, seek bar, time and chords agree |
| Pause / resume keeps the timeline | pass | resume plays at once, no reset |
| Framewise and Viterbi rows labelled | pass | "Decoded: HMM / Viterbi, fixed lag of 4 frames", "Framewise: network alone, no decoder" |
| Same axis, same recording, same playhead | pass | one timeline, one playhead, all rows from the same posteriors |
| Framewise not altered | pass | `tests/test_demo.py`: single-frame runs kept |
| Missing output reported | pass | a truncated row is refused with a message |
| Pipeline lists implemented stages in order | pass | 8 stages, values from the run |
| No unimplemented stage claimed | pass | no "no chord", no beat tracking or chroma; Viterbi "not used" at 0.28 s |
| Reference only when verified | pass | GuitarSet lead-sheet annotations for the samples, none for uploads |
| Reference distinct from predictions | pass | separate row, different style, "reference labels, not model output" |
| A supported upload is analysed | pass | WAV, FLAC, Ogg, MP3 (Phase 4); live: an M4A decoded by the browser and streamed, chords to the end of the file |
| Invalid file gives a clear message | pass | a broken WAV ("this browser cannot decode it as audio"), too large or too long files |
| Failed analysis shows no stale result | pass | the page is cleared first; a server error stops the session and shows its message (`tests/test_demo_server.py`: refused audio, busy server, too long) |
| New file replaces old results | pass | every pick clears the page and closes the running session; an older load drops its result |
| No flicker or rerender during playback | pass | 0 blocks rebuilt during playback |
| Repeated playback | pass | played to the end twice, then again from the start (with the delay) |
| No console errors in the normal flow | pass, with a note | none from the page with the demo server; refused uploads log their 400 response, and the browser pane requests `/favicon.ico` itself |

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
| Same recording and run (Phase 2): every row of the three samples starts at 0, has no gap, ends at the audio's duration; the page refuses a row that does not (checked with a truncated row) | all rows pass; the truncated one is refused |
| Playing to the end with the delay off | the decoder flushes: the whole row shown, last chord, status "Ended" |
| Three sample picks 5 ms apart | the last pick wins; nothing of the others shown; look-ahead and delay settings kept |
| Phase 3: reference toggle off by default; on, its blocks match the JSON; reference chord and active block vs the JSON every ~16 ms over 4 s of playback, with the decoded row's on-time check in the same run | 0 mismatches, 0 early, 0 ms late, no block rebuilt |
| Phase 3: a sample without reference; pipeline at 0.28 s and 0.46 s | checkbox disabled with the "not available" note; the Viterbi stage says "not used" at 0.28 s and gives lag 2 at 0.46 s |
| Console | no page errors; the browser pane itself requests `/favicon.ico` (404) although the page declares no icon file |

## Open questions

- **Half a frame.** The delay includes half a frame so that chords are on time with segments cut halfway
  between frames, as the WCSR cuts them. Whether the causal features' frame time convention would put
  the frame elsewhere was not checked; the prototype's delay (`play_along.required_delay`) has no half
  frame.
- **Precomputed mode only: seeking is not a restart.** Without the server, after a seek the page shows
  the decisions of the continuous run; live mode restarts the session instead.
- **The margin.** 0.15 s covers a local server with room to spare; a remote one adds its round trip. The
  panel's measurement says when it is not enough; the margin could then follow the measurement.
- **Precomputed decision delay.** It counts one frame more than live streaming needs (see "Delay"
  above): never early, a frame late.
- **Hosting.** Live decoding needs `scripts/demo_server.py` (Python, the model, the configs and the
  Billboard counts) on a machine with about 10 ms of CPU per frame per listener; without it the page
  falls back to the precomputed samples.
- **Narrow blocks** hide their name (tooltip, and the large chord when it is the current one); the spec
  asks for text labels always.

## Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_demo.py tests/test_demo_server.py tests/test_play_along.py
PYTHONPATH=src python scripts/demo_export.py 00_Funk2-108-Eb_comp 00_Rock3-148-C_comp 00_SS3-98-C_comp
PYTHONPATH=src python scripts/demo_server.py          # http://localhost:8000, live decoding
```
