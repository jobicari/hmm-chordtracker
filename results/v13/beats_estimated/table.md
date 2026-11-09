Lag-8 fixed-lag Viterbi on V11 seed 1's validation posteriors, RWC-P and GuitarSet.
Cells: majmin / sevenths % (chord changes per minute). Reference changes per minute: rwc 30, guitarset 27, mean 28.

Chosen on the mean sevenths of the two sets:

| beats | width | p_on | p_off | RWC-P | GuitarSet | mean |
|---|---:|---:|---:|---:|---:|---:|
| none |  |  | 0.98 | 81.98 / 72.13 (35) | 77.94 / 70.99 (31) | 79.96 / 71.56 (32) | frozen
| none |  |  | 0.95 | 82.01 / 72.14 (35) | 77.97 / 71.02 (31) | 79.99 / 71.58 (32) | best constant
| annotated | 1 | 0.95 | 0.995 | 82.12 / 72.21 (35) | 78.19 / 71.25 (31) | 80.16 / 71.73 (32) |
| downbeats | 1 | 0.3 | 0.998 | 82.49 / 72.62 (35) | 79.39 / 72.25 (32) | 80.94 / 72.43 (33) |
| librosa | 0 | 0.9 | 0.98 | 82.00 / 72.12 (35) | 78.04 / 71.13 (31) | 80.02 / 71.62 (32) |
| madmom | 1 | 0.95 | 0.99 | 82.10 / 72.21 (35) | 78.12 / 71.18 (31) | 80.11 / 71.70 (32) |
| madmom downbeats | 1 | 0.3 | 0.998 | 82.48 / 72.60 (35) | 78.38 / 71.34 (32) | 80.43 / 71.97 (33) |

Chosen on one set, judged on the other (sevenths %, change against the best constant chosen the same way):

| beats | chosen on | judged on | setting (width, p_on, p_off) | sevenths | best constant | change |
|---|---|---|---|---:|---:|---:|
| annotated | rwc | guitarset | 1, 0.7, 0.98 | 70.98 | 71.02 (p 0.95) | -0.04 |
| annotated | guitarset | rwc | 1, 0.95, 0.995 | 72.21 | 72.09 (p 0.995) | +0.11 |
| downbeats | rwc | guitarset | 1, 0.3, 0.995 | 72.10 | 71.02 (p 0.95) | +1.08 |
| downbeats | guitarset | rwc | 1, 0.3, 0.998 | 72.62 | 72.09 (p 0.995) | +0.53 |
| librosa | rwc | guitarset | 1, 0.7, 0.98 | 70.74 | 71.02 (p 0.95) | -0.29 |
| librosa | guitarset | rwc | 0, 0.95, 0.99 | 72.11 | 72.09 (p 0.995) | +0.01 |
| madmom | rwc | guitarset | 1, 0.5, 0.98 | 70.81 | 71.02 (p 0.95) | -0.21 |
| madmom | guitarset | rwc | 1, 0.95, 0.99 | 72.21 | 72.09 (p 0.995) | +0.12 |
| madmom downbeats | rwc | guitarset | 1, 0.3, 0.995 | 71.31 | 71.02 (p 0.95) | +0.29 |
| madmom downbeats | guitarset | rwc | 0, 0.3, 0.998 | 71.54 | 72.09 (p 0.995) | -0.55 |

Beats per minute, mean over the songs: annotated 122, downbeats 31, librosa 118, madmom 117, madmom downbeats 29.
F-measure against the annotations (beats against beats, downbeats against downbeats; mean over the songs): librosa rwc 0.874, guitarset 0.534, mean 0.619; madmom rwc 0.993, guitarset 0.788, mean 0.839; madmom downbeats rwc 0.989, guitarset 0.566, mean 0.672.
