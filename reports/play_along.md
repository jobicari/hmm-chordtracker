# Listening prototype: chords live with delayed playback

## Question

V7.2 and V7.3 give the accuracy for each total look-ahead and a live chain that keeps up with the
music. They cannot say how much playback delay a musician accepts, or how early a chord should appear
to be playable. **A prototype that plays a recording with delayed playback, analyses it as it arrives
and shows the chords live lets those settings be judged by ear**, next to a reference annotation.

## Summary

The prototype works end to end with Spotify as the source: the chords follow the music, with the
quality the validation numbers predict. Findings from a first listening session (10 Beatles songs):

1. **Only the longest look-ahead is steady enough.** At 0.28 s the network alone changes chord too
   often to follow; at 0.65 s (input look-ahead 2, fixed-lag Viterbi lag 4) the chords are stable.
   This matches V7.2: about 60 against 35 chord changes per minute on validation.
2. **A long lead was preferred.** The session ended at a 1.0 s lead (chords shown a second before they
   sound) and an annotation nudge of +0.75 s, for 1.74 s of playback delay in total.
3. **Three weaknesses stand out**: the 24 major/minor chords are too few for the Beatles (sevenths and
   other qualities show up as their triads); the chords are not stable enough at short delays; and a
   strong bass line (Drive My Car) throws the detection off, likely because the bass notes outweigh the
   rest of the spectrum.
4. **One unexplained reading.** Help! was read as its relative minor on a first run and correctly after
   a reset. The network's memory, the running level and the start tuning all depend on how the
   analysis started; which one caused it was not checked.

## Setup

`scripts/play_along.py` runs on the Mac:

    source      Spotify -> Loopback -> VB-Cable A (input device), or a file read at real-time pace
    analysis    V7.3's live chain in its own process: StreamingFeatures (C2 grid) -> the V7.2 network with
                input look-ahead 2 (results/v7_2/c2_la2_seed0) -> per operating point: the network alone
                (0.28 s), fixed-lag Viterbi lag 2 (0.46 s), lag 4 (0.65 s)
    playback    the input, delayed by look-ahead + one frame of margin + lead + extra, on the output device
    window      the system's chord now and next; the Isophonics annotation of the song picked in the
                drop-down (now and next); a settings panel with the keys
    reference   Isophonics Beatles chord annotations (180 songs, data/beatles/, not in the repo); their
                clock starts when the input rises out of silence and can be nudged

Keys (first session): 1 / 2 / 3 operating point; up / down extra delay; left / right lead; `[` / `]` annotation earlier
/ later; space reset. Every change is logged (`data/play_along/`, not in the repo).

The logic that does not need sound devices (`src/chordtracker/play_along.py`: chord names, the
annotation carousel, the onset, what the window shows at a given moment, the delay each setting needs,
the analysis) is covered by `tests/test_play_along.py`.

## What live testing found and fixed

Each was reproduced on the Mac before it was fixed.

| Symptom | Cause | Fix |
|---|---|---|
| Nothing happens | VB-Cable A received the music at about -47 dBFS (the Mac's volume applies to the cable), below the onset threshold | volume up; onset threshold at -100 dBFS; input level in the panel |
| Nothing happens from PyCharm's terminal | the program received only silence from the cable | started from the Claude app's terminal, which has audio input access |
| Window frozen on "..." | the window's clock followed the output stream | clock from the input |
| Choppy playback, then silence | the analysis shared the audio threads' interpreter lock, and playback re-synced at 20 ms of drift | analysis in its own process; playback continuous, re-sync only past 0.25 s |
| Choppy playback, 77 to 99 % silence | VB-Cable fills large input buffers (4,096 samples) mostly with zeros | small input buffers (480 samples) |
| Tuning warning on silence | the analysis started when the song was picked, not when it played | the analysis starts at the onset |
| Nudge keys backwards | `]` moved the annotation earlier | `]` later, `[` earlier |
| Leaked semaphores on quit | the analysis process and its queues were not closed | closed on quit |

## What the next version should look at

- **A richer chord vocabulary**: sevenths, N (no chord) and more, which the Beatles annotations use
  throughout; open since V5.
- **Steadier chords at shorter delays**: the hold rule (V7.3) on top of the fixed-lag Viterbi, or a
  longer decoder lag, scored on validation and checked by ear.
- **Robustness to strong bass lines**: why a busy bass part pulls the detection (Drive My Car), and
  whether a per-register balance in the input or more bass-heavy training data fixes it.
- **The start of a song**: whether a minor/relative-minor mix-up like Help!'s comes from the network's
  memory, the running level or the start tuning.

## Second session: V11's network with sevenths (V12)

Since V12 the prototype runs V11's frozen network (`results/v11/pool_kd1_seed1`, `configs/v11.yaml`):
vocabulary A (maj, min, 7, maj7, min7 on 12 roots), prior power 0.25, the Billboard change matrix, and
V11's four decoder points on keys 1 to 4: the network alone (0.28 s) and fixed-lag Viterbi at lag 2,
4 and 8 (0.46, 0.65 and 1.02 s), each at the p_self frozen on validation. The chord names include the
sevenths (G7, Cmaj7, Am7), and the annotation is reduced to the same vocabulary as it is scored
(mir_eval's sevenths: D:9 is D7, C:maj6 is C); a chord outside it (dim, sus4, aug) is shown as it is,
in parentheses. The demo website keeps V7.2's network and its three points.

Findings from a listening session on Beatles songs:

1. **The longest look-ahead works best.** Lag 8 (1.02 s, p_self 0.98) is clearly the steadiest and the
   most often right, as on validation, where it is the best of the four points on all three sets
   (73.15 % mean sevenths against 71.87 % for the network alone).
2. **The remaining errors are plausible** for what the network was trained on (19 h of labelled audio,
   little of it with the Beatles' range of chords) and for a five-quality vocabulary.
3. **A moving bass under a held chord often throws the chord off**: walking bass lines, passing notes
   and inversions. The training labels ignore the bass (C:maj/3 is trained as C:maj) while the input
   starts at C2, so a new bass note changes the input much as a new chord would. This is the first
   session's strong-bass finding, now that the sevenths are in place.

One more fix from this round, reproduced with `--dry-run` before it was fixed:

| Symptom | Cause | Fix |
|---|---|---|
| "..." for about 0.4 s, 10 s into every song | the start tuning (estimated on the first 10 s) recomputes the harmonic part of those 10 s in one step, so the analysis falls behind the playback delay once | the last labelled chord stays shown until the late frame is labelled; the features are unchanged |

`--dry-run SECONDS` (with `--file`) runs a file at real-time pace through the same analysis process,
with no window and no sound, and prints the analysis time per input block and, per point at its own
playback delay, how many 30 ms updates found their frame not yet labelled. On 30 s of an RWC-P
validation song at 48 kHz: 2.5 ms of analysis per 21.3 ms block on average (one block of 0.39 s, the
tuning step), no dropouts, and no update without a chord after the first label.

## Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_play_along.py
PYTHONPATH=src python scripts/play_along.py --list-devices
PYTHONPATH=src python scripts/play_along.py --input-device "VB-Cable A" --output-device "MacBook Pro Speakers"
PYTHONPATH=src python scripts/play_along.py --file data/rwc/RWC-P/RWC_P037.wav --dry-run 30
PYTHONPATH=src python scripts/play_along.py --model v14 --file data/rwc/RWC-P/RWC_P037.wav
PYTHONPATH=src python scripts/play_along.py --model v14 --silence-gate --input-device "VB-Cable A"
```

`--model v14` runs V14's network with an N class (results/v14/n_seed0, git-ignored; `--run` points at
another folder) at its best decoder setting on validation (N weight x0.1, chord -> N rate x0.5), with
the silence gate off, so the N shown is the network's own. V14 did not pass its checks
([reports/v14.md](v14.md)): it calls N on some chords. The default stays V11 with the silence gate.

The Isophonics annotations go in `data/beatles/chordlab/` (from isophonics.net, "The Beatles
Annotations"); the audio comes from your own player.
