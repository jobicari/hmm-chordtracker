# V10: validation notes

Working notes for V10, validation only (RWC-P val 20 takes, GuitarSet val 60, ChoCo val 47; never
pooled, "mean" is the mean of the three sets). Test was not run. Every model is vocabulary A (60
classes, no N), and the reference is V8's vocabulary A seed 0 unless a step says otherwise.

Cells are aligned WCSR, majmin / sevenths %. The four decoder points are V8's: network alone (frame
argmax, 0.28 s), and fixed-lag Viterbi at lag 2, 4 and 8 (0.46, 0.65, 1.02 s) with the Billboard
matrix and prior power 0.25. For each point, p_self is the value on configs/v8.yaml's grid with the
best mean sevenths. Commands: `scripts/run_v10.py` (decoder-side steps and the `points` scorer) and
`scripts/train_v10.py` (retrains).

V8 vocabulary A on validation, the reference for every step:

| V8 vocabulary A | point | p_self | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|---:|
| seed 0 | network alone (0.28 s) |  | 79.67 / 68.53 | 76.39 / 68.06 | 85.22 / 75.72 | 80.43 / 70.77 |
| seed 0 | lag 2 (0.46 s) | 0.5 | 80.44 / 69.65 | 77.30 / 68.93 | 85.91 / 75.78 | 81.21 / 71.45 |
| seed 0 | lag 4 (0.65 s) | 0.95 | 81.17 / 70.27 | 77.91 / 69.41 | 86.40 / 76.20 | 81.83 / 71.96 |
| seed 0 | lag 8 (1.02 s) | 0.95 | 81.37 / 70.80 | 78.33 / 69.73 | 86.51 / 76.65 | 82.07 / 72.39 |
| seed 1 (frozen) | network alone (0.28 s) |  | 79.94 / 69.75 | 76.12 / 68.64 | 84.92 / 75.10 | 80.33 / 71.16 |
| seed 1 (frozen) | lag 2 (0.46 s) | 0.5 | 80.79 / 70.79 | 76.56 / 69.75 | 85.48 / 74.99 | 80.94 / 71.84 |
| seed 1 (frozen) | lag 4 (0.65 s) | 0.8 | 81.38 / 71.50 | 77.12 / 70.33 | 86.04 / 75.52 | 81.51 / 72.45 |
| seed 1 (frozen) | lag 8 (1.02 s) | 0.98 | 81.77 / 72.14 | 77.70 / 70.87 | 86.49 / 75.96 | 81.99 / 72.99 |

The two V8 seeds are 0.60 sevenths apart at lag 8 (72.39 and 72.99). Any change smaller than that is
within seed noise.

## Summary

| step | change | best mean sevenths at lag 8 | against | verdict |
|---|---|---:|---|---|
| 1 | seed ensemble (seeds 0 + 1, geometric) | 73.32 | 72.99 (seed 1) | small gain, costs two networks |
| 2 | per-quality posterior offsets | 72.41 to 73.28 | 72.39 / 72.99 | null: does not survive the guards |
| 3 | tuning estimate at 12 bins (issue #11) | 72.40 | 72.39 | fix kept, no score change on val |
| 4 | level gate for no-chord | see step 4 | | gains only on reference-N seconds |
| 5 | early stopping on sevenths | 72.39 | 72.39 | null: same epoch as V8 |
| 6 | root / quality / bass heads | 72.46 | 72.39 | null: +0.07, follow-ups not run |
| 7 | gain, tilt and noise augmentation | 71.09 | 72.39 | worse: -1.30 |

## Step 1: seed ensemble

Setup: the network posteriors of V8 seed 0 and seed 1 combined, then decoded. Two combinations:
geometric (the normalised mean of the log posteriors) and arithmetic (the mean). Each was also
decoded at prior power 0.

| system, prior power 0.25 | point | p_self | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|---:|
| geometric ensemble | network alone (0.28 s) |  | 80.76 / 70.39 | 76.82 / 69.17 | 85.68 / 76.30 | 81.09 / 71.95 |
| geometric ensemble | lag 2 (0.46 s) | 0.3 | 81.32 / 71.21 | 77.54 / 70.42 | 86.22 / 76.01 | 81.69 / 72.55 |
| geometric ensemble | lag 4 (0.65 s) | 0.8 | 82.09 / 71.98 | 77.81 / 70.66 | 86.77 / 76.38 | 82.22 / 73.01 |
| geometric ensemble | lag 8 (1.02 s) | 0.8 | 82.26 / 72.63 | 78.07 / 70.70 | 86.86 / 76.63 | 82.40 / 73.32 |
| arithmetic ensemble | network alone (0.28 s) |  | 80.73 / 70.33 | 76.87 / 69.19 | 85.67 / 76.29 | 81.09 / 71.93 |
| arithmetic ensemble | lag 2 (0.46 s) | 0.5 | 81.46 / 71.24 | 77.33 / 70.23 | 86.31 / 75.87 | 81.70 / 72.45 |
| arithmetic ensemble | lag 4 (0.65 s) | 0.5 | 81.88 / 71.87 | 77.86 / 70.65 | 86.56 / 76.24 | 82.10 / 72.92 |
| arithmetic ensemble | lag 8 (1.02 s) | 0.95 | 82.27 / 72.37 | 78.32 / 70.92 | 86.94 / 76.50 | 82.51 / 73.26 |
| two-seed average of the scores | lag 8 (1.02 s) |  | 81.57 / 71.47 | 78.02 / 70.30 | 86.50 / 76.30 | 82.03 / 72.69 |

Verdict: the ensemble gains most for the network alone. There it scores 71.95 sevenths, +0.79 on seed
1 and +0.98 on the two-seed average. At lag 8 it scores 73.32, +0.33 on seed 1 and +0.63 on the
average. Geometric and arithmetic are within 0.06. The gain is real but small, and it doubles the
network cost per frame. Prior power 0 is lower at every point (seed 0 lag 8: 72.06 against 72.39).

## Step 2: per-quality offsets

Setup: one log offset per quality (maj fixed at 0; min, 7, maj7, min7 on a grid from -1.5 to +1.0),
added to the posterior before decoding. A coordinate search fits the offsets to the mean sevenths of
the fitting sets, at lag 8 or for the network alone. There are two guards against fitting
validation noise: (a) fit on one seed, judge on the other; (b) on seed 1, fit on two sets and judge
on the third.

| guard (a): fitted on | judged on | lag 8, mean sevenths before | after |
|---|---|---:|---:|
| seed 0 | seed 0 (in-sample) | 72.39 | 72.53 |
| seed 1 | seed 0 | 72.39 | 72.41 |
| seed 1 | seed 1 (in-sample) | 72.99 | 73.28 |
| seed 0 | seed 1 | 72.99 | 72.94 |

| guard (b), seed 1: held-out set | V8, lag 8 | offsets, lag 8 | V8, network alone | offsets, network alone |
|---|---:|---:|---:|---:|
| RWC-P | 81.77 / 72.14 | 81.28 / 66.04 | 79.94 / 69.75 | 79.20 / 64.05 |
| GuitarSet | 77.70 / 70.87 | 77.97 / 69.95 | 76.12 / 68.64 | 75.89 / 67.62 |
| ChoCo | 86.49 / 75.96 | 86.26 / 75.69 | 84.92 / 75.10 | 84.84 / 74.89 |

Verdict: null. In-sample the offsets add 0.14 to 0.29, but across seeds they add +0.02 and -0.05. On
a held-out set they lose sevenths every time, by 6.1 points on RWC-P: fitted without RWC-P, the maj7
offset is -1.5, and RWC-P has the most maj7 (5.0 % of its reference frames; seed 0 predicts 2.6 %
for the network alone). The offsets fit the quality mix of the fitting sets, not a bias of the
model. Applied to the ensemble they add at most +0.18 at lag 8 (73.32 to 73.50). Not adopted.

## Step 3: tuning estimate (issue #11)

Setup: `features_v7_1.start_tuning` estimated the tuning at 24 bins per octave, which wraps at +/-25
cents, so a recording tuned near a quarter tone could be shifted the wrong way. V10 estimates at 12
bins per octave (+/-50 cents, no wrap at 25). Over the 200 RWC-P and GuitarSet train and val songs,
15 estimates move by more than 5 cents; the median move is 0.5 cents and the largest 52.5
(guitarset/02_Rock1-130-A_comp, train: -24.5 to +28.0). Four val songs move, all GuitarSet, by 6 to
7.5 cents. They were re-cached and scored with V8's models (not retrained).

| features | lag 8 (1.02 s), mean | network alone, mean |
|---|---:|---:|
| seed 0, cached (24 bins) | 82.07 / 72.39 | 80.43 / 70.77 |
| seed 0, retuned (12 bins) | 82.08 / 72.40 | 80.43 / 70.77 |
| seed 1, cached (24 bins) | 81.99 / 72.99 | 80.33 / 71.16 |
| seed 1, retuned (12 bins) | 82.00 / 73.00 | 80.33 / 71.16 |

Verdict: the fix is kept for correctness. On val it changes the mean by at most 0.03; the songs it
matters most for are train songs, which would only count after a retrain.

## Step 4: level gate for no-chord

Setup: after decoding, a frame becomes N (no chord) where its level is below a threshold. The level
is the mean of the frame's cached log-CQT column, which is relative to the song's running level and
0 in digital silence. Seed 1, network alone and lag 8 at p_self 0.98. "With N" scores reference N as
mir_eval does (V8 never outputs N, so it gets every reference-N second wrong). "N-excluded" leaves
reference N out, so the gate can only lose there. GuitarSet has no reference N.

| seed 1, lag 8, threshold | RWC-P with N | ChoCo with N | GuitarSet | RWC-P N-excl. | ChoCo N-excl. | false N s (RWC-P / GS / ChoCo) |
|---:|---:|---:|---:|---:|---:|---|
| 0 (V8) | 81.77 / 72.14 | 86.49 / 75.96 | 77.70 / 70.87 | 83.79 / 73.99 | 88.72 / 77.97 | 0 / 0 / 0 |
| 0.005 | 83.37 / 73.80 | 87.47 / 76.96 | 77.46 / 70.64 | 83.71 / 73.91 | 88.40 / 77.64 | 10 / 11 / 37 |
| 0.01 | 83.29 / 73.73 | 87.42 / 76.91 | 77.28 / 70.49 | 83.62 / 73.83 | 88.25 / 77.48 | 16 / 19 / 54 |
| 0.02 | 83.08 / 73.54 | 87.22 / 76.72 | 76.89 / 70.17 | 83.39 / 73.62 | 87.98 / 77.21 | 29 / 31 / 79 |
| 0.05 | 82.57 / 73.17 | 86.76 / 76.29 | 75.05 / 68.69 | 82.85 / 73.22 | 87.42 / 76.69 | 58 / 85 / 131 |

At 0.005 the gate finds 75 of RWC-P's 107 reference-N seconds and 113 of ChoCo's 219. The network
alone behaves the same way (RWC-P with N 69.75 to 71.41 sevenths, ChoCo 75.10 to 76.11, GuitarSet
68.64 to 68.44).

Verdict: with N scored, the lowest threshold gains 1.66 sevenths on RWC-P and 1.00 on ChoCo. All of
that comes from reference-N seconds, which V8 cannot label. Where the reference is a chord, the gate
only loses (0.08 to 0.33 at 0.005, more as the threshold rises), and GuitarSet loses 0.23. A higher
threshold is worse everywhere. It is a scoring and display choice, not a better chord reading; if
kept, 0.005 is the only sensible threshold.

## Step 5: early stopping on sevenths

Setup: retrain A seed 0 with V8's recipe, stopping on the mean val sevenths frame score instead of
the mean frame accuracy (`train.early_stop: sevenths`), patience 5.

Result: best epoch 14 of 19, the same epoch V8 seed 0 stops at on frame accuracy. The weights are the
same and so is every table cell (lag 8: 82.07 / 72.39).

Verdict: null. On this run the two criteria pick the same epoch.

## Step 6: root, quality and bass heads

Setup: retrain A seed 0 with three extra linear heads on the shared hidden layer: root (12), quality
(5) and bass (12, the lowest note of the Harte label). Loss = chord cross-entropy + 0.5 * (root +
quality + bass cross-entropy); pitch-shift augmentation moves root and bass with the chord. Decoding
uses the 60-class chord head only, so the decoders are V8's. Early stopping on frame accuracy, as V8.
Best epoch 18 of 23 (V8 seed 0: 14 of 19).

| A seed 0 | point | p_self | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|---:|
| V8 | network alone (0.28 s) |  | 79.67 / 68.53 | 76.39 / 68.06 | 85.22 / 75.72 | 80.43 / 70.77 |
| V8 | lag 2 (0.46 s) | 0.5 | 80.44 / 69.65 | 77.30 / 68.93 | 85.91 / 75.78 | 81.21 / 71.45 |
| V8 | lag 4 (0.65 s) | 0.95 | 81.17 / 70.27 | 77.91 / 69.41 | 86.40 / 76.20 | 81.83 / 71.96 |
| V8 | lag 8 (1.02 s) | 0.95 | 81.37 / 70.80 | 78.33 / 69.73 | 86.51 / 76.65 | 82.07 / 72.39 |
| heads | network alone (0.28 s) |  | 80.07 / 70.07 | 75.66 / 67.98 | 85.11 / 75.09 | 80.28 / 71.05 |
| heads | lag 2 (0.46 s) | 0.5 | 81.24 / 70.86 | 76.11 / 68.74 | 85.78 / 74.81 | 81.04 / 71.47 |
| heads | lag 4 (0.65 s) | 0.9 | 81.74 / 71.42 | 76.57 / 69.18 | 86.28 / 75.25 | 81.53 / 71.95 |
| heads | lag 8 (1.02 s) | 0.98 | 82.03 / 72.07 | 77.24 / 69.69 | 86.56 / 75.61 | 81.94 / 72.46 |

Verdict: null. At lag 8 the mean sevenths gain is +0.07, well below the +0.5 threshold, so the
planned follow-ups (heads with early stopping on sevenths, then a seed 1) were not run. The mean
hides opposite moves per set: RWC-P +1.27 sevenths, GuitarSet -0.04, ChoCo -1.04. Mean majmin falls
at every point (-0.13 at lag 8). For the network alone sevenths rises by 0.28, also within the
0.60-point seed gap.

## Step 7: input augmentation

Setup: retrain A seed 0 with three augmentations of each log-CQT training excerpt, on top of V8's
pitch shift (`train.augment`, default off). All three are drawn per excerpt from a separate
generator, so the excerpts, their order and their shifts stay V8's.
- gain: uniform in +/-3 dB
- linear spectral tilt: uniform in +/-6 dB from the lowest CQT row to the highest
- noise: Gaussian with standard deviation 0.05 on every value, clipped at 0

Gain and tilt are applied inside the log compression (x' = log(1 + (e^x - 1) g)). Early stopping on
frame accuracy. Best epoch 12 of 17.

| A seed 0 | point | p_self | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|---:|
| V8 | network alone (0.28 s) |  | 79.67 / 68.53 | 76.39 / 68.06 | 85.22 / 75.72 | 80.43 / 70.77 |
| V8 | lag 8 (1.02 s) | 0.95 | 81.37 / 70.80 | 78.33 / 69.73 | 86.51 / 76.65 | 82.07 / 72.39 |
| augmented | network alone (0.28 s) |  | 78.38 / 67.07 | 75.69 / 67.89 | 84.50 / 74.25 | 79.52 / 69.74 |
| augmented | lag 2 (0.46 s) | 0.3 | 78.94 / 67.92 | 75.90 / 68.46 | 84.88 / 74.11 | 79.90 / 70.16 |
| augmented | lag 4 (0.65 s) | 0.8 | 79.57 / 68.65 | 76.52 / 69.01 | 85.33 / 74.42 | 80.47 / 70.69 |
| augmented | lag 8 (1.02 s) | 0.9 | 79.78 / 69.25 | 77.04 / 69.35 | 85.61 / 74.66 | 80.81 / 71.09 |

Verdict: worse. At lag 8 the mean falls by 1.26 majmin and 1.30 sevenths, more than the seed gap.
Sevenths falls on every set (RWC-P -1.55, GuitarSet -0.38, ChoCo -1.99) and at every point. It
stays off by default; weaker settings were not tried.

## Where V10 leaves the model

None of the retrains (steps 5 to 7) beats V8 seed 0 by more than seed noise. Of the decoder-side
changes, only the seed ensemble (step 1) gains, and only by about half the seed gap at lag 8. The
tuning fix (step 3) stays as a correctness fix. V8 vocabulary A seed 1 stays the frozen model, and
test was not run.
