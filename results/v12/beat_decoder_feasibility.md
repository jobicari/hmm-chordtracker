# A beat-aware causal decoder: feasibility

Decode-only study on validation (RWC-P and GuitarSet), no training. Script: `scripts/v12_beats.py`;
numbers: `results/v12/beats/table.md`.

## What V1 did, and what is proposed now

**V1 (reports/v1.md, src/chordtracker/beats.py)** changed the *features and the time axis*: the
23 ms chroma frames were averaged over each beat, and the whole decoder (one template decision or
one HMM step per beat) ran on the beat grid. The beats came from `librosa.beat.beat_track` on the
full audio file, so they were offline and used the whole song. Beat averaging replaced the HMM's
smoothing rather than adding to it, and V7 later found frame-level Viterbi at least as good as
the beat HMM (77.04 % vs 76.57 % mean WCSR). Beats were then dropped.

**Proposed now**: keep everything frame-level (the network, its 93 ms frames, the posteriors) and use
beats only *inside the decoder's transition*: the self-transition probability depends on the frame,
p_on near a beat and p_off elsewhere (p_off > p_on), so a chord change is cheap near a beat and dear
between beats. The fixed-lag Viterbi stays as it is otherwise (lag 8, 1.02 s of total look-ahead).
With p_on = p_off it is exactly today's decoder (tested).

## Trackers available without installing anything

| Environment | Package | Beats online | Downbeats online |
|---|---|---|---|
| hmm-chordtracker | librosa 1.0 (`beat.beat_track`, `beat.plp`) | no: whole-signal dynamic programming and a centred tempogram | no |
| hmm-chordtracker | madmom, BeatNet, essentia, aubio | not installed | not installed |
| chord-refs (separate env) | madmom 0.17 | yes: `RNNBeatProcessor(online=True)` + `DBNBeatTrackingProcessor(online=True)` (forward algorithm, frame by frame) | no: `RNNDownBeatProcessor` is bidirectional, `DBNDownBeatTrackingProcessor` has no online mode |

So causal beats exist (madmom online, in a separate env); causal downbeats do not, off the shelf.
BeatNet (causal joint beat and downbeat tracking with a particle filter) would, but is not installed.

## Latency budget

At lag 8, the label of frame t is decided when frame t + 8 has been read, 0.74 s after frame t (1.02 s
of total look-ahead with the input look-ahead and the harmonic frame). Whether frame t is near a beat
is needed only at that moment, so a tracker may confirm a beat up to about 0.7 s after it happened.
madmom's online beat tracker reports beats as they occur, well within that. A downbeat estimate needs
the bar's phase, which a causal tracker can only settle after a few bars; until then the decoder
would run as today (p_on = p_off).

## Upper bound on validation

V11's frozen student (seed 1), its saved validation posteriors, lag 8. Mean of RWC-P (20 songs) and
GuitarSet (60), majmin / sevenths %, chord changes per minute in brackets (reference: 28). Grid:
p_on 0.3 to 0.95, p_off 0.98 to 0.998, width 0 or 1 frame around the beat; the controls include
constant p_self up to 0.998, so a gain is not just more smoothing.

| beats | setting (width, p_on, p_off) | RWC-P | GuitarSet | mean |
|---|---|---:|---:|---:|
| none, frozen | p_self 0.98 | 81.98 / 72.13 (35) | 77.94 / 70.99 (31) | 79.96 / 71.56 (32) |
| none, best constant | p_self 0.95 | 82.01 / 72.14 (35) | 77.97 / 71.02 (31) | 79.99 / 71.58 (32) |
| every annotated beat | 1, 0.95, 0.995 | 82.12 / 72.21 (35) | 78.19 / 71.25 (31) | 80.16 / 71.73 (32) |
| annotated downbeats only | 1, 0.3, 0.998 | 82.49 / 72.62 (35) | 79.39 / 72.25 (32) | **80.94 / 72.43** (33) |
| librosa beats (offline, estimated) | 0, 0.9, 0.98 | 82.00 / 72.12 (35) | 78.04 / 71.13 (31) | 80.02 / 71.62 (32) |

The settings above are chosen on the same two sets. Chosen on one set and judged on the other,
against the best constant p_self chosen the same way (sevenths, points):

| beats | chosen on RWC-P, judged on GuitarSet | chosen on GuitarSet, judged on RWC-P |
|---|---:|---:|
| every annotated beat | -0.04 | +0.11 |
| annotated downbeats | **+1.08** | **+0.53** |
| librosa beats | -0.29 | +0.01 |

Annotated: RWC-P's AIST beat annotations and GuitarSet's beat_position, with the bar position for
downbeats; ChoCo's beats are not in data/. librosa: `beat_track` on the full audio, offline.

**Reading.** Beats as such carry almost nothing: about two beats a second, so "near a beat" covers
most chord changes anyway, and the oracle gains 0.15 points. Downbeats are the informative part:
chords change mostly on bar lines, and making changes dear everywhere else (p_off 0.998) but cheap on
the downbeat (p_on 0.3) gains +0.85 points of mean sevenths over the best constant p_self (+0.87 over the frozen lag 8) with the true bar lines, and holds when
the setting is chosen on the other set (+0.53, +1.08). That is larger than any decoder-side change
since V8 (the V10 ensemble: +0.33 at lag 8). It is an upper bound: true downbeats, offline.

## Recommendation

**No-go for beats; a conditional go for downbeats.** A beat-only decoder is not worth building: its
oracle gain is within seed noise. A downbeat-aware decoder is worth one more cheap step before any
real-time work: measure how much of the +0.85 survives *estimated* downbeats.

1. Offline estimated downbeats first (madmom's `RNNDownBeatProcessor` + `DBNDownBeatTrackingProcessor`
   in the chord-refs env, run by hand on the 80 validation songs, times saved as files and read by
   `v12_beats.py`). If less than about half the gain survives, stop.
2. If it survives, the causal version: madmom's online beat tracker plus a bar-phase estimate, which
   could come from the chord decoder itself (a joint state of chord and position in the bar, where a
   change is likely at position 1). That is a design question for its own version, and it needs madmom
   (or BeatNet) in the main environment, which needs approval.
3. Check by ear: changes on bar lines should also look steadier at lag 8, and possibly let a shorter
   lag reach today's lag-8 steadiness.
