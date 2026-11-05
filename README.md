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
| V3 | Learned chord-change prior (train labels, or 730 Billboard songs) vs V2's flat self-loop; null result on val, so V3 keeps the V2 decoder | 50.1 % | 57.5 % (V2 decoder, not rerun) | [reports/v3.md](reports/v3.md) |
| V4 | Learned emission (logistic regression on V2 beat chroma, fitted on train, divided by the train class prior) with V3's Billboard change prior at full strength | n/a (Viterbi only) | 62.3 % (p_self 0.5) | [reports/v4.md](reports/v4.md) |
| V5 | CNN acoustic model on log-CQT patches (pitch-shift augmentation, trained on RWC-P + GuitarSet train), V4's decoder | n/a (Viterbi only) | 72.9 % (p_self 0.4); RWC-P test 79.0 % | [reports/v5.md](reports/v5.md) |
| V5.1 | No-chord (N) class on V5 (25 states); null result on val (76.6% vs 76.98% mean, best decoder 76.7%), so V5.1 keeps V5 | n/a | 72.9 % (V5, not rerun); RWC-P 79.0 % | [reports/v5_1.md](reports/v5_1.md) |
| V5.2 | Weight decay (5 strengths) and AAM synthetic songs (1,000, three shares) on V5; null result on val (best AAM 76.99% vs 76.98% mean, seed gap 0.9), so V5.2 keeps V5 | n/a | 72.9 % (V5, not rerun); RWC-P 79.0 % | [reports/v5_2.md](reports/v5_2.md) |
| V6 | CRNN: V5's conv front end + a GRU over the frames (bidirectional or causal), same data and decoder; on val the network alone gains (76.5% vs 74.9% mean of two seeds) but with the HMM V6 ties V5 (77.0% vs 76.5%, seed gap 0.8) | n/a | 74.7 % (p_self 0.4, one seed); RWC-P test 80.0 % | [reports/v6.md](reports/v6.md) |
| V7 | Decoders that only use past frames, per CRNN frame (no beats), on V6's causal model (not retrained); on val frames match or beat beats (offline frame Viterbi 77.0% vs beat Viterbi 76.6%), the HMM at lag 0 does not beat the network alone, and once the decoder's delay is counted (on-time WCSR) no lag beats the network alone; the input features already need ~1.6 s of future audio | 71.4 % (network alone, lag 0, best on-time WCSR on validation); RWC-P 79.6 % | 72.1 % (fixed-lag Viterbi, lag 0.37 s); offline frame Viterbi 72.4 %, RWC-P 80.1 % | [reports/v7.md](reports/v7.md) |
| V7.1 | Input features that use no future audio (causal harmonic median, CQT bins delayed by half their kernel, running level, tuning from the first 10 s, window ending at the frame), causal CRNN retrained; on validation the network alone loses 4.6 points per label (71.5% vs 76.1%, the loss sits in the first 0.5 s after chord changes) but scores 68.4% end to end (V7: 25.0%) | 68.2 % (network alone, lag 0, aligned; end-to-end 66.1 %); RWC-P 73.7 % | 70.4 % (offline frame Viterbi); RWC-P 76.1 % | [reports/v7_1.md](reports/v7_1.md) |
| V7.2 | Input look-ahead of 0 to 4 frames (window of frame t ends at t + a) and CQT from C2, for playback delayed by the total look-ahead; on validation 2 frames of input look-ahead (0.28 s in total) bring the network alone to 75.7% (V7.1 71.5%, V7 with 1.6 s 76.1%), a fixed-lag Viterbi adds up to 1.4 points and halves the flicker, C2 ties F#0 | 71.9 % (network alone, 0.28 s, C2); RWC-P 78.7 % | 71.3 % (fixed-lag Viterbi, 0.65 s); RWC-P 80.1 % | [reports/v7_2.md](reports/v7_2.md) |
| V7.3 | The causal features computed block by block (2,048 samples at a time), as a live system receives audio; on validation the chords match the cached features within 0.05 points at 0.28 / 0.46 / 0.65 s, and the whole live chain takes 7 ms per 93 ms block; a hold rule against flicker ties the fixed-lag Viterbi | n/a (no test run: same chords as V7.2) | n/a | [reports/v7_3.md](reports/v7_3.md) |
| Listening prototype | The live chain with delayed playback and Isophonics Beatles chords shown alongside, fed from Spotify through a virtual cable; by ear, only the 0.65 s look-ahead is steady enough, a 1 s lead was preferred, and the 24-chord vocabulary, chord stability and strong bass lines are the next problems | n/a | n/a | [reports/play_along.md](reports/play_along.md) |

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
| `src/chordtracker/transitions.py` | V3: transposition-invariant chord-change counts, learned transition matrix (used by V4) |
| `scripts/count_billboard.py` | V3: change counts from the Billboard labels (ChoCo JAMS) |
| `scripts/v3_cache_val.py`, `scripts/v3_ablate_transitions.py`, `scripts/v3_ablate_min_duration.py` | V3: the val ablations behind reports/v3.md |
| `src/chordtracker/learned_emission.py` | V4: training rows (12 transpositions), logistic regression fit, posteriors, saved model |
| `configs/v4.yaml` | V4: extends v2.yaml; emission C, prior division, Billboard tau, p_self |
| `scripts/run_v4.py` | V4: fit on train + val sweep, or test once with the saved model |
| `scripts/v4_cache_beats.py`, `scripts/v4_ablate_emissions.py`, `scripts/v4_ablate_priors.py` | V4: the val ablations behind reports/v4.md |
| `src/chordtracker/features_v5.py` | V5: log-CQT (168 bins, hop 2048) and its cache in data/processed/v5_logcqt/ |
| `src/chordtracker/data_v5.py` | V5: RWC-P and GuitarSet songs, maj/min targets, pitch-shifted patches |
| `src/chordtracker/model_v5.py` | V5: the CNN acoustic model |
| `src/chordtracker/train_v5.py` | V5: training loop, early stopping, save/load |
| `src/chordtracker/acoustic_hmm.py` | V5: frame posteriors to beats, training prior, V4 decoder |
| `configs/v5.yaml` | V5: extends v2.yaml; CQT, training, Billboard prior, p_self |
| `scripts/run_v5.py` | V5: train + val sweep, or test once with the saved model |
| `scripts/run_v5_baseline.py` | V5: V4 refitted on V5's training data (val baseline) |
| `src/chordtracker/data_v5_1.py` | V5.1: 25-class targets (N = 24), pitch shift keeps N |
| `src/chordtracker/model_v5_1.py` | V5.1: V5's CNN with a 25th output |
| `src/chordtracker/train_v5_1.py` | V5.1: V5's training loop with 25-class targets |
| `src/chordtracker/acoustic_hmm_v5_1.py` | V5.1: 25-state prior, change prior with N, N seconds found / missed / false |
| `configs/v5_1.yaml` | V5.1: V5's settings with 25 states |
| `scripts/run_v5_1.py` | V5.1: train + val sweep |
| `scripts/v5_1_ablate_decoder.py` | V5.1: decoder ablation on val (N transition rate x N prior) |
| `src/chordtracker/data_aam.py` | V5.2: AAM beat labels -> chord segments, AAM takes read from the log-CQT cache |
| `src/chordtracker/train_v5_2.py` | V5.2: V5's training with AdamW weight decay and AAM frames mixed into each epoch |
| `configs/v5_2.yaml` | V5.2: V5's settings + weight_decay and aam_ratio (both 0) |
| `scripts/fetch_aam.py` | V5.2: AAM mixes read one by one from the Zenodo zip, CRC-checked, cached, deleted |
| `scripts/v5_2_check_aam.py` | V5.2: AAM data check (label shares, template accuracy by label shift) |
| `scripts/run_v5_2.py` | V5.2: train + val sweep, with --weight-decay, --seed, --aam-ratio, --aam-songs |
| `src/chordtracker/data_v6.py` | V6: training excerpts (128 frames + 7 of context each side), one shift per excerpt, per-epoch tiling |
| `src/chordtracker/model_v6.py` | V6: the CRNN (V5's conv blocks per frame, 1 x 1 conv, embedding, GRU), whole-song posteriors |
| `src/chordtracker/train_v6.py` | V6: training loop on excerpts, early stopping, save/load |
| `configs/v6.yaml` | V6: V5's settings + excerpt_frames, batch_excerpts, bidirectional |
| `scripts/run_v6.py` | V6: train + val (argmax per beat and HMM sweep), with --seed, --causal, --batch-excerpts; test once with --freeze |
| `scripts/v6_v5_argmax.py`, `scripts/v6_context_check.py` | V6: V5's network alone on val; val frame accuracy with the GRU restarted every n frames |
| `src/chordtracker/online_hmm.py` | V7: forward filtering and fixed-lag Viterbi, streaming classes and whole-song functions |
| `src/chordtracker/streaming_crnn.py` | V7: the causal CRNN fed one CQT column at a time (same posteriors as the whole-song pass) |
| `configs/v7.yaml` | V7: per-frame p_self grid, lags, the input's look-ahead, the choices frozen on val |
| `scripts/run_v7.py` | V7: five decoders on a causal V6 run, aligned / on-time / end-to-end WCSR; test once with --freeze |
| `scripts/v7_plot.py`, `scripts/v7_speed.py` | V7: the latency figure (validation); time per real-time frame on the CPU |
| `src/chordtracker/features_v7_1.py` | V7.1: causal log-CQT (past-only harmonic median, delayed CQT bins, running level, start tuning), its cache |
| `src/chordtracker/data_v7_1.py`, `train_v7_1.py` | V7.1: excerpts whose window ends at the frame; training the causal CRNN on them |
| `src/chordtracker/streaming_v7_1.py` | V7.1: the causal CRNN fed one column at a time, no wait |
| `configs/v7_1.yaml` | V7.1: causal feature settings (median 9, tuning 10 s), V6's training, V7's decoders, frozen choices |
| `scripts/run_v7_1.py` | V7.1: train + decode on validation (V7's systems and scores), --median-frames; test once with --freeze |
| `scripts/v7_1_tuning.py`, `v7_1_breakdown.py`, `v7_1_plot.py` | V7.1: training-song tuning; accuracy after chord changes and at song starts; latency figure |
| `src/chordtracker/data_v7_2.py`, `train_v7_2.py`, `streaming_v7_2.py` | V7.2: window with an input look-ahead, the C2 grid's cache, training, column-by-column inference |
| `configs/v7_2.yaml` | V7.2: V7.1's settings + input_lookahead, the CQT grid, the frozen curve points |
| `scripts/run_v7_2.py` | V7.2: train + decode one cell (--cqt-from, --input-lookahead, --seed); test once over the frozen points |
| `scripts/v7_2_curve.py`, `v7_2_breakdown.py` | V7.2: best aligned WCSR per total look-ahead + figure; accuracy after chord changes by cell |
| `src/chordtracker/streaming_features.py` | V7.3: the causal log-CQT block by block (harmonic part, direct CQT, running level, start tuning); the same computation on a whole file |
| `src/chordtracker/hold.py` | V7.3: the hold rule against flicker |
| `scripts/v7_3_check.py`, `v7_3_speed.py`, `v7_3_hold.py` | V7.3: same chords as the cache on validation; time per block of the live chain; the hold rule on validation |
| `src/chordtracker/play_along.py`, `scripts/play_along.py` | Listening prototype: what to show when, the live analysis in its own process; the player (input device or file, delayed playback, window, keys, log) |
| `src/chordtracker/demo.py`, `scripts/demo_export.py`, `demo/` | Demo website: outputs of GuitarSet validation takes exported with the prototype's chain; a static page that plays one and reveals its chords as the live system would, framewise vs fixed-lag Viterbi ([reports/demo.md](reports/demo.md)) |
| `splits/guitarset.json` | the fixed song-level split (seed 0), tracked |
| `splits/rwc_p.json` | RWC-P 60 / 20 / 20 split by artist |
| `data/` | GuitarSet, RWC-P and everything generated from them (git-ignored) |
| `results/<version>/<split>/` | `summary.json` and `per_song.csv` of each run |
| `figures/<version>/val/` | the saved report figures (val only) |
| `reports/<version>.md` | each version's write-up: question, setup, val analysis, test score, findings |

## Setup

Python 3.12 in the conda env `hmm-chordtracker`; dependencies are in `pyproject.toml` / `uv.lock`.
GuitarSet goes in `data/guitarset/` (`annotation/` and `audio_mono-mic_comp/`). The manifest and
`.lab` files in `data/processed/guitarset/` are generated from it by `chordtracker.data.data.prepare`.

RWC Popular Music (RWC-P), 100 songs, audio and chord labels under CC BY-NC 4.0 (Zenodo 18656623,
rwc-music/rwc-annotations), placed in `data/rwc/` (git-ignored). Used from V5; split by artist:
`splits/rwc_p.json`.

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
| Emission (posterior) | P(chord \| frame) = softmax over the 25 states of beta * (templates @ chroma), beta = 1 |
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

**Terminology.** In the textbook HMM the emission is the likelihood P(evidence | chord). The
emission models in this repo output the posterior P(chord | evidence) instead; the decoder
(`viterbi_discriminative`) divides it by a state prior P(chord) to get a scaled likelihood,
P(evidence | chord) / P(evidence), the standard hybrid neural-network/HMM convention. V0 to V3 use
a uniform prior (the division changes nothing), V4 the train class frequencies.

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
