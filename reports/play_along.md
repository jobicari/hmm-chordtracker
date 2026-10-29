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

Keys: 1 / 2 / 3 operating point; up / down extra delay; left / right lead; `[` / `]` annotation earlier
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

## Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_play_along.py
PYTHONPATH=src python scripts/play_along.py --list-devices
PYTHONPATH=src python scripts/play_along.py --input-device "VB-Cable A" --output-device "MacBook Pro Speakers"
```

The Isophonics annotations go in `data/beatles/chordlab/` (from isophonics.net, "The Beatles
Annotations"); the audio comes from your own player.
