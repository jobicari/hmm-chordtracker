V10's heads model (seed 0), validation, decode only. Chord head posteriors recomputed here match the saved ones within 4.9e-06.

Majmin / sevenths % (chord changes per minute); p_self chosen per row on the mean sevenths of the three sets. Last column: decoded changes inside held RWC-P chords per minute (those near a MIDI bass change).

| system | p_self | RWC-P | GuitarSet | ChoCo | mean | RWC-P changes inside held chords / min (near a MIDI bass change) |
|---|---:|---:|---:|---:|---:|---|
| network alone |  | 80.07 / 70.07 (74) | 75.66 / 67.98 (61) | 85.11 / 75.09 (59) | 80.28 / 71.05 (62) | 27.46 (14.68) |
| lag 8, gamma 0 | 0.98 | 82.03 / 72.07 (35) | 77.24 / 69.69 (33) | 86.56 / 75.61 (29) | 81.94 / 72.46 (32) | 6.24 (3.17) |
| lag 8, gamma 0.5 | 0.98 | 81.99 / 72.06 (34) | 77.24 / 69.67 (32) | 86.53 / 75.56 (29) | 81.92 / 72.43 (32) | 6.15 (3.04) |
| lag 8, gamma 0.9 | 0.98 | 81.90 / 72.01 (34) | 77.19 / 69.60 (33) | 86.40 / 75.44 (29) | 81.83 / 72.35 (31) | 6.20 (2.96) |
| lag 8, gamma 0.99 | 0.98 | 81.84 / 71.96 (34) | 77.16 / 69.57 (32) | 86.25 / 75.32 (29) | 81.75 / 72.28 (31) | 6.15 (2.95) |

Inversion-aware scores, majmin_inv / sevenths_inv % (ChoCo: the 22 Beatles validation songs matched to data/beatles):

| system | labels | RWC-P | GuitarSet | ChoCo (Beatles) |
|---|---|---:|---:|---:|
| network alone | root position | 74.85 / 65.56 | 75.66 / 67.98 | 78.51 / 70.20 |
| network alone | slash | 75.21 / 65.96 | 74.51 / 67.26 | 78.85 / 70.58 |
| lag 8, gamma 0 | root position | 76.76 / 67.55 | 77.24 / 69.69 | 79.86 / 71.16 |
| lag 8, gamma 0 | slash | 77.01 / 67.83 | 76.22 / 69.14 | 80.08 / 71.42 |
| lag 8, gamma 0.5 | root position | 76.74 / 67.54 | 77.24 / 69.67 | 79.83 / 71.13 |
| lag 8, gamma 0.5 | slash | 76.96 / 67.81 | 76.21 / 69.12 | 80.02 / 71.37 |
| lag 8, gamma 0.9 | root position | 76.63 / 67.48 | 77.19 / 69.60 | 79.76 / 71.08 |
| lag 8, gamma 0.9 | slash | 76.81 / 67.71 | 76.17 / 69.05 | 79.89 / 71.26 |
| lag 8, gamma 0.99 | root position | 76.58 / 67.44 | 77.16 / 69.57 | 79.64 / 70.99 |
| lag 8, gamma 0.99 | slash | 76.76 / 67.65 | 76.09 / 68.99 | 79.74 / 71.14 |
