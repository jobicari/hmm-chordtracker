# External references: BTC, madmom and Chordino against V6

**Question.** How does the frozen V6 compare with published, pretrained chord recognisers on the same
songs, splits and metric? Nothing here is trained or tuned: each tool runs as shipped.

**Summary of the results.** No tool clearly beats V6 on test. On RWC-P, BTC is 1.64 points ahead, all of
it from its N class: with reference N left out, V6 and BTC are level (82.07 and 81.82). On GuitarSet,
madmom is 1.42 ahead and BTC 1.88 behind, but both gaps changed sign between validation and test, by
about 3 points. Chordino is behind on GuitarSet by 6.81 points.

## Results

WCSR in percent. V6 is the frozen model of `reports/v6.md` (`results/v6/bidir_seed1`, HMM at p_self 0.4).

**Headline: pooled WCSR, every scored second counts.**

| | RWC-P val | RWC-P test | GuitarSet val | GuitarSet test |
|---|---:|---:|---:|---:|
| V6 | 78.23 | 80.00 | 76.62 | 74.69 |
| BTC | 82.56 | 81.64 | 77.34 | 72.81 |
| madmom | | | 74.87 | 76.11 |
| Chordino | 76.37 | 79.54 | 71.03 | 67.88 |

**N-excluded: the same score with reference N left out.** V6 has 24 classes and no N, so every second of
reference N counts against it. GuitarSet has no reference N, so its scores do not change.

| | RWC-P val | RWC-P test |
|---|---:|---:|
| V6 | 80.17 | 82.07 |
| BTC | 82.72 | 81.82 |
| Chordino | 77.22 | 80.32 |

**Seconds of reference N each tool got right.**

| | RWC-P val (of 107.5) | RWC-P test (of 120.4) |
|---|---:|---:|
| V6 | 0.0 | 0.0 |
| BTC | 81.8 | 90.1 |
| Chordino | 45.4 | 59.5 |

## How the scores are computed

- **Songs and audio.** `run_v5.evaluation_sets(split)`: 20 RWC-P songs and 60 GuitarSet takes per split, the
  files V6 reads, with its references. Every tool scored V6's exact seconds.
- **WCSR.** `evaluate.score` with the majmin metric, then `evaluate.pooled`: matched seconds / scored seconds
  over a whole set, as in `reports/v6.md`. Example: 15 of 20 s and 5 of 10 s give 20 / 30 = 66.7%, and a long
  song weighs more than a short one. `mean_song_wcsr` in `summary.json` is the mean of the per-song scores.
- **Tool labels.** Read through `data_v5.target`, as V5 and V6 targets are: C:7 and C:maj7 count as C:maj,
  A:min7 and C:dim as A:min and C:min. N stays N. A chord with no third would count as N; none was output.
- **N-excluded.** (matched - matched on reference N) / (scored - reference N), both parts from `evaluate.score`.
  Example, RWC-P test, BTC: 3,905.3 s matched minus 90.1 s on N, over 4,783.4 - 120.4 s, gives 81.82%. V6
  stores durations rather than predictions; with no N class it matches exactly 0 s of reference N.
- **Pairs.** BTC and Chordino on both sets. madmom on GuitarSet only: its models were trained on RWC Pop,
  so its RWC-P numbers would be leaked. BTC was trained on neither set.
- **Test.** Run once, after the validation run, and only pooled numbers were read.

## Limitations

- V6 is one seed, and its seed, decoder and p_self were chosen on the validation songs, which flatters its
  validation score. The tools chose nothing, so test is the fairer comparison. V6's seeds differ by about a point.
- BTC does not normalise level. On one song per set, its log-CQT mean is -1.79 on RWC-P and -4.19 on GuitarSet,
  against -2.23 in its checkpoint, so it may be at a disadvantage on the quieter GuitarSet recordings. Not corrected.
- Every tool reads the whole song, so these numbers say nothing about real-time use.

## Tools

| Tool | Version | Licence |
|---|---|---|
| BTC, `btc_model.pt`, 25 classes | [BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19) `2682317be668032e6e4b269ded36adaa2ad57df0` | MIT |
| madmom, `CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor` | 0.17.dev0, [madmom](https://github.com/CPJKU/madmom) `27f032e8947204902c675e5e341a3faf5dc86dae` | code BSD, models CC BY-NC-SA 4.0 |
| Chordino, `nnls-chroma:chordino`, simplechord output, default parameters | NNLS Chroma from Vamp Plugin Pack 2.0, arm64 | GPL-2.0 |
| vamp, Python host | 1.1.0 | MIT |

Python 3.10.22, torch 2.14.1, numpy 2.2.6, scipy 1.15.3, librosa 0.11.0, mir_eval 0.8.2, jams 0.3.5.
`nnls-chroma.dylib` sha256: `22b001129db5beec65abb8ac6952be6ed2cd2d809aec3d8ebb50d2495afc2dd6`.

BTC's code predates PyYAML 6 and numpy 1.24, so `score_refs.py` reads its config with `yaml.safe_load` and
restores `np.float`. Its clone is unmodified. Runtime on test on an Apple Silicon Mac: BTC 27 s, Chordino 164 s,
madmom 510 s.

## Reproduce

Tools and weights stay outside the repo, in `CHORD_REFS_DIR` (default `~/chordtracker_v5_scratch/refs`).

```bash
mamba create -y -n chord-refs -c conda-forge python=3.10 pip setuptools wheel
conda activate chord-refs
pip install "numpy<2.3" scipy cython
pip install torch librosa "mir_eval==0.8.2" jams pretty_midi pyyaml soundfile
pip install --no-build-isolation vamp
pip install --no-build-isolation "git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae"

mkdir -p "$CHORD_REFS_DIR/plugins"
git clone https://github.com/jayg996/BTC-ISMIR19 "$CHORD_REFS_DIR/btc"
git -C "$CHORD_REFS_DIR/btc" checkout 2682317be668032e6e4b269ded36adaa2ad57df0
# install Vamp Plugin Pack 2.0 (vamp-plugins.org/pack.html), then copy nnls-chroma.dylib from
# ~/Library/Audio/Plug-Ins/Vamp to "$CHORD_REFS_DIR/plugins"

python scripts/refs/check_setup.py                                        # one validation song per tool
python scripts/refs/score_refs.py --tools btc madmom chordino --splits val
python scripts/refs/rescore_n.py val
python scripts/refs/score_refs.py --tools btc madmom chordino --splits test     # once
python scripts/refs/rescore_n.py test
```

`score_refs.py` appends to `per_song.csv`, and the last row of a song wins; saved `.lab` files are reused.
