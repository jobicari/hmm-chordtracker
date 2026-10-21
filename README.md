# HMM chord tracker

A controlled comparison for audio chord recognition: does an HMM with Viterbi decoding beat
independent frame-by-frame classification when the features, chord vocabulary and songs are held
constant? Built version by version (V0, V1, ...); each version changes one thing and is compared
with the previous one on the same split and metric.

**Rule for every version: test is scored, never inspected.** Choices and investigations (which
p_self, why a take fails, what the next version should fix) are made on the validation songs only.
The test songs are decoded once per version to report a score, and no figure or error analysis is
ever made from them. See [Evaluation protocol](#evaluation-protocol).

## Results

Test WCSR (majmin, pooled over the 60 test takes), run once with the setting chosen on val. Each
version's report has the question, the method, the val analysis and the limitations.

| Version | What it tests | test WCSR, framewise | test WCSR, HMM | Report |
|---|---|---|---|---|
| V0 | Viterbi smoothing vs frame-by-frame argmax, same template emission | 41.8 % | 53.7 % (p_self 0.95) | [reports/v0.md](reports/v0.md) |
| V1 | Beat-synchronous chroma (mean per beat) vs V0's frames, same emission and decoders | 48.8 % | 53.3 % (p_self 0.05) | [reports/v1.md](reports/v1.md) |
| V2 | Better chroma on V1's beats: low-band floor (body resonance), log compression, bass chroma (E1..B3) in the template score | 50.1 % | 57.5 % (p_self 0.06) | [reports/v2.md](reports/v2.md) |

## Layout

| Path | What it holds |
|---|---|
| `configs/v0.yaml` | every V0 setting: audio, chroma, templates, emission, decoder grid, metric |
| `configs/v1.yaml`, `configs/v2.yaml` | the V1 and V2 settings, each choice next to its val numbers |
| `src/chordtracker/data/` | `ChordAnnotation`, the GuitarSet adapter, manifest and split handling |
| `src/chordtracker/features.py` | audio loading and `ChromaCQTExtractor` |
| `src/chordtracker/templates.py`, `emission.py` | 25 maj/min/N templates, `TemplateEmission` (softmax posteriors) |
| `src/chordtracker/decode.py` | `FramewiseDecoder` (argmax) and `ViterbiDecoder(p_self)` |
| `src/chordtracker/evaluate.py` | frames to segments, duration-weighted WCSR, label policy |
| `src/chordtracker/experiment.py` | per-take evidence, p_self sweep, choice, result files |
| `src/chordtracker/visualize.py` | plotting functions: data in, matplotlib Figure out; nothing decodes or scores |
| `src/chordtracker/report.py` | `v0_figures` (all V0 figures, val only), `save_figures`, `results_tables` (report tables) |
| `scripts/run_v0.py` | command line for a V0 run on one split |
| `notebooks/V0_figures.ipynb`, `scripts/plot_v0.py` | the V0 figures and tables, inline in a notebook or saved from the command line |
| `src/chordtracker/beats.py` | V1: beat tracking, per-beat pooling, beat-edge segments, beat-level p_self sweep |
| `scripts/run_v1.py` | command line for a V1 run on one split (same protocol as run_v0.py) |
| `notebooks/V1_val_chroma_check.ipynb` | the G#/A investigation behind V1 (val only) |
| `src/chordtracker/features_v2.py` | V2: own CQT, per-bin floor, log compression, treble and bass chroma |
| `src/chordtracker/bass.py` | V2: bass term of the emission (root of each chord), V2 take evidence |
| `scripts/run_v2.py` | command line for a V2 run on one split (same protocol as run_v1.py) |
| `splits/guitarset.json` | the fixed song-level split (seed 0), tracked |
| `data/` | GuitarSet and everything generated from it (git-ignored) |
| `results/<version>/<split>/` | `summary.json` and `per_song.csv` of each run |
| `figures/<version>/val/` | the saved report figures (val only) |
| `reports/<version>.md` | each version's write-up: question, setup, val analysis, test score, findings |

## Setup

Python 3.12 in the conda env `hmm-chordtracker`; dependencies are in `pyproject.toml` / `uv.lock`.
GuitarSet goes in `data/guitarset/` (`annotation/` and `audio_mono-mic_comp/`). The manifest and
`.lab` files in `data/processed/guitarset/` are generated from it by `chordtracker.data.data.prepare`.

## Tests

```bash
python -m pytest              # fast suite, no audio needed
python -m pytest -m slow      # needs GuitarSet; checks the pipeline on one train take
```

Tests check behaviour (shapes, Viterbi at p_self = 1/25 equals framewise, Viterbi smooths, the
label policy), not numbers copied from a notebook.

## V0 pipeline

| Step | Setting |
|---|---|
| Audio | mono, 22050 Hz, harmonic part only (`librosa.effects.harmonic`, margin 4) |
| Chroma | `chroma_cqt`, hop 512 (23.2 ms), threshold 0.25, tuning estimated per file, per-frame max norm |
| Templates | 12 major + 12 minor triads (1 on the chord's pitch classes) + N (all 0.25) |
| Emission | P(chord \| frame) = softmax over the 25 states of beta * (templates @ chroma), beta = 1 |
| Framewise | argmax of each frame's posteriors |
| Viterbi | `librosa.sequence.viterbi_discriminative` with `transition_loop(25, p_self)` |
| Metric | `mir_eval.chord.majmin`, duration-weighted (WCSR), per take and pooled over the split |

## Label policy

V0 can only output major triads, minor triads and N, while the GuitarSet lead sheets use richer
chords. With the majmin metric, each reference chord is compared through its triad:

| Reference | Compared as | Counted |
|---|---|---|
| C:maj, A:min, N | itself | scored |
| C:7, C:maj7, C:maj6 | C:maj | scored, **reduced** |
| A:min7, A:min6 | A:min | scored, **reduced** |
| C:maj/3 | C:maj | scored, **reduced** (bass ignored) |
| B:hdim7, D:sus4, X | none | **excluded** from the score |

So a V0 WCSR answers "did the model find the right triad?", not "the written chord". Every result
reports `reduced_s` and `excluded_s` next to the WCSR, so the size of that simplification is
visible. A later version with a sevenths vocabulary will be scored with `mir_eval.chord.sevenths`.

## Evaluation protocol

The split is by song group (style + progression), never by frame: 60 comp takes each in train,
val and test, with every take of a progression in the same split.

1. **Choose on val.** `python scripts/run_v0.py --split val` decodes the val takes with framewise
   and with Viterbi at every p_self of `decoder.p_self_grid`, and picks the p_self with the best
   pooled WCSR (a tie goes to the smaller p_self).
2. **Freeze.** Commit the code and the config. From here on, nothing about the version changes.
3. **Test once.** `python scripts/run_v<N>.py --split test --p-self <chosen>` runs framewise and
   Viterbi at that single p_self. The script refuses `--split test` without `--p-self`, so test
   cannot be swept.
4. **Investigate on val only.** Every figure and every "why does it fail" question uses val takes.
   From test only the pooled numbers are reported. `report.v0_figures` refuses a test take and
   writes only to `figures/<version>/val/`.

**Why test is run only once.** The test score is meant to estimate how the model does on
progressions it was never tuned on. That only holds while no decision has been made by looking at
test. If a test result led to changing p_self, the templates, the features or the label policy,
the test set would have become a second validation set, the reported score would be optimistic,
and no untouched data would be left to say by how much. Re-running test with nothing changed is
harmless (the run is deterministic and gives the same numbers). Each new version (V1, ...) makes
its own choices on val and then runs test once in the same way.

**Why test is never inspected either.** Looking at where a version fails on test, then designing
the next version to fix it, tunes that next version to the test songs just as surely as choosing
p_self on them would, only indirectly. So the error analysis that motivates V1 is done on val, and
test stays a set of songs nothing was ever decided from.

## Figures

All V0 figures come from the val split, as the rule above requires.

- **Notebook:** open `notebooks/V0_figures.ipynb` with the `hmm-chordtracker` interpreter and run
  all cells. The figures show inline, the next cell prints the report tables, the last one saves
  the PNGs.
- **Command line:**

  ```bash
  python scripts/plot_v0.py --p-self 0.95 --take 00_SS3-84-Bb_comp --take 04_Funk2-119-G_comp
  ```

  writes `figures/v0/val/{sweep, per_take, groups, durations, take_<song_id>}.png` and prints the
  tables.

The sweep, per-take and style figures read `results/v0/val/`; the take and duration figures
recompute the val posteriors from the audio (about 80 s). Nothing is rescored, and `--p-self` must
be the value already chosen on val. The `chordtracker` package must be importable: installed in
the env (`pip install -e .`) or with `PYTHONPATH=src`.
