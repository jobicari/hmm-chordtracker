# V15: dim, aug and hdim7 on top of V14's N model, validation

Validation only (RWC-P val, GuitarSet val, ChoCo val; mean = the mean of the three sets), the features V14 n_seed0 is scored on. Lag 8 is V14 n_seed0's demo point: fixed-lag Viterbi lag 8, p_self 0.98, N weight 0.1, chord -> N rate x0.5, prior power 0.25; V15 runs divide by the fine-tune's target mix or by the natural train prior. Reference N is scored (V14's "with N"). majmin and sevenths leave dim, aug, dim7 and hdim7 references out, so the new classes can only cost them; tetrads scores every quality exactly, mirex counts an output sharing three notes with the reference as right.

## majmin %

| system | point | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|
| V14 n_seed0 | network | 80.92 | 75.62 | 86.39 | 80.98 |
| V14 n_seed0 | lag 8 | 83.34 | 78.36 | 87.98 | 83.22 |
| V15 seed 0 (62.5 % replay) | network | 81.41 | 74.67 | 86.33 | 80.80 |
| V15 seed 0 (62.5 % replay) | lag 8, mix prior | 83.93 | 76.80 | 87.43 | 82.72 |
| V15 seed 0 (62.5 % replay) | lag 8, natural prior | 83.94 | 76.73 | 87.27 | 82.65 |
| V15 seed 0 (62.5 % replay), best sevenths | network | 81.43 | 74.79 | 86.15 | 80.79 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, mix prior | 83.43 | 78.25 | 87.28 | 82.99 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, natural prior | 83.47 | 78.23 | 87.25 | 82.98 |
| V15 seed 0 (50 % replay) | network | 81.04 | 74.64 | 86.25 | 80.64 |
| V15 seed 0 (50 % replay) | lag 8, mix prior | 83.49 | 77.12 | 87.34 | 82.65 |
| V15 seed 0 (50 % replay) | lag 8, natural prior | 83.33 | 76.96 | 87.22 | 82.50 |
| V15 seed 0 (50 % replay), best sevenths | network | 81.18 | 75.41 | 86.11 | 80.90 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, mix prior | 83.78 | 77.78 | 87.33 | 82.96 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, natural prior | 83.70 | 77.84 | 87.14 | 82.90 |
| V15 seed 0 (focus only) | network | 80.00 | 74.32 | 85.31 | 79.88 |
| V15 seed 0 (focus only) | lag 8, mix prior | 82.87 | 76.54 | 86.86 | 82.09 |
| V15 seed 0 (focus only) | lag 8, natural prior | 82.42 | 76.20 | 86.34 | 81.65 |
| V15 seed 0 (focus only), best sevenths | network | 80.92 | 75.62 | 86.39 | 80.98 |
| V15 seed 0 (focus only), best sevenths | lag 8, mix prior | 83.38 | 78.21 | 87.95 | 83.18 |
| V15 seed 0 (focus only), best sevenths | lag 8, natural prior | 83.34 | 78.36 | 87.98 | 83.23 |

## sevenths %

| system | point | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|
| V14 n_seed0 | network | 70.39 | 67.04 | 76.76 | 71.40 |
| V14 n_seed0 | lag 8 | 73.32 | 69.48 | 77.52 | 73.44 |
| V15 seed 0 (62.5 % replay) | network | 70.82 | 66.43 | 77.07 | 71.44 |
| V15 seed 0 (62.5 % replay) | lag 8, mix prior | 73.64 | 68.74 | 77.64 | 73.34 |
| V15 seed 0 (62.5 % replay) | lag 8, natural prior | 73.71 | 68.75 | 77.28 | 73.25 |
| V15 seed 0 (62.5 % replay), best sevenths | network | 71.64 | 67.19 | 76.24 | 71.69 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, mix prior | 74.22 | 70.06 | 76.73 | 73.67 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, natural prior | 74.26 | 70.11 | 76.39 | 73.59 |
| V15 seed 0 (50 % replay) | network | 70.57 | 65.99 | 76.69 | 71.08 |
| V15 seed 0 (50 % replay) | lag 8, mix prior | 73.41 | 68.02 | 77.19 | 72.87 |
| V15 seed 0 (50 % replay) | lag 8, natural prior | 73.24 | 67.98 | 76.92 | 72.71 |
| V15 seed 0 (50 % replay), best sevenths | network | 71.79 | 67.01 | 75.95 | 71.58 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, mix prior | 75.11 | 68.98 | 76.25 | 73.44 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, natural prior | 74.83 | 69.25 | 76.03 | 73.37 |
| V15 seed 0 (focus only) | network | 69.59 | 65.87 | 74.71 | 70.05 |
| V15 seed 0 (focus only) | lag 8, mix prior | 73.28 | 67.83 | 75.74 | 72.28 |
| V15 seed 0 (focus only) | lag 8, natural prior | 72.51 | 67.63 | 74.91 | 71.68 |
| V15 seed 0 (focus only), best sevenths | network | 70.39 | 67.04 | 76.76 | 71.40 |
| V15 seed 0 (focus only), best sevenths | lag 8, mix prior | 73.04 | 69.49 | 77.80 | 73.44 |
| V15 seed 0 (focus only), best sevenths | lag 8, natural prior | 73.32 | 69.48 | 77.52 | 73.44 |

## tetrads %

| system | point | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|
| V14 n_seed0 | network | 61.32 | 63.85 | 68.24 | 64.47 |
| V14 n_seed0 | lag 8 | 63.86 | 66.18 | 68.91 | 66.32 |
| V15 seed 0 (62.5 % replay) | network | 62.14 | 64.10 | 68.68 | 64.97 |
| V15 seed 0 (62.5 % replay) | lag 8, mix prior | 64.61 | 66.34 | 69.19 | 66.71 |
| V15 seed 0 (62.5 % replay) | lag 8, natural prior | 64.77 | 66.49 | 68.91 | 66.72 |
| V15 seed 0 (62.5 % replay), best sevenths | network | 62.40 | 63.99 | 67.78 | 64.72 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, mix prior | 64.65 | 66.73 | 68.21 | 66.53 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, natural prior | 64.69 | 66.78 | 67.91 | 66.46 |
| V15 seed 0 (50 % replay) | network | 62.00 | 63.87 | 68.37 | 64.75 |
| V15 seed 0 (50 % replay) | lag 8, mix prior | 64.48 | 65.83 | 68.81 | 66.37 |
| V15 seed 0 (50 % replay) | lag 8, natural prior | 64.41 | 66.06 | 68.61 | 66.36 |
| V15 seed 0 (50 % replay), best sevenths | network | 62.94 | 64.46 | 67.65 | 65.02 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, mix prior | 65.82 | 66.40 | 67.90 | 66.71 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, natural prior | 65.66 | 66.91 | 67.73 | 66.77 |
| V15 seed 0 (focus only) | network | 61.18 | 63.86 | 66.59 | 63.88 |
| V15 seed 0 (focus only) | lag 8, mix prior | 64.40 | 65.76 | 67.51 | 65.89 |
| V15 seed 0 (focus only) | lag 8, natural prior | 63.83 | 65.79 | 66.83 | 65.48 |
| V15 seed 0 (focus only), best sevenths | network | 61.32 | 63.85 | 68.24 | 64.47 |
| V15 seed 0 (focus only), best sevenths | lag 8, mix prior | 63.63 | 66.18 | 69.17 | 66.32 |
| V15 seed 0 (focus only), best sevenths | lag 8, natural prior | 63.87 | 66.18 | 68.91 | 66.32 |

## mirex %

| system | point | RWC-P | GuitarSet | ChoCo | mean |
|---|---|---:|---:|---:|---:|
| V14 n_seed0 | network | 79.44 | 74.57 | 83.45 | 79.15 |
| V14 n_seed0 | lag 8 | 81.97 | 77.73 | 84.81 | 81.50 |
| V15 seed 0 (62.5 % replay) | network | 79.93 | 73.76 | 83.58 | 79.09 |
| V15 seed 0 (62.5 % replay) | lag 8, mix prior | 82.15 | 76.14 | 84.52 | 80.94 |
| V15 seed 0 (62.5 % replay) | lag 8, natural prior | 82.26 | 76.13 | 84.49 | 80.96 |
| V15 seed 0 (62.5 % replay), best sevenths | network | 80.10 | 73.19 | 83.45 | 78.91 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, mix prior | 81.92 | 76.61 | 84.50 | 81.01 |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, natural prior | 82.06 | 76.67 | 84.50 | 81.08 |
| V15 seed 0 (50 % replay) | network | 79.67 | 73.77 | 83.36 | 78.93 |
| V15 seed 0 (50 % replay) | lag 8, mix prior | 81.65 | 76.35 | 84.36 | 80.78 |
| V15 seed 0 (50 % replay) | lag 8, natural prior | 81.63 | 76.60 | 84.31 | 80.85 |
| V15 seed 0 (50 % replay), best sevenths | network | 80.40 | 74.30 | 83.49 | 79.39 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, mix prior | 82.62 | 76.74 | 84.59 | 81.32 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, natural prior | 82.73 | 77.02 | 84.55 | 81.43 |
| V15 seed 0 (focus only) | network | 79.55 | 73.91 | 82.66 | 78.70 |
| V15 seed 0 (focus only) | lag 8, mix prior | 81.71 | 76.16 | 83.98 | 80.62 |
| V15 seed 0 (focus only) | lag 8, natural prior | 81.63 | 76.43 | 83.69 | 80.58 |
| V15 seed 0 (focus only), best sevenths | network | 79.44 | 74.57 | 83.45 | 79.15 |
| V15 seed 0 (focus only), best sevenths | lag 8, mix prior | 81.80 | 77.41 | 84.79 | 81.33 |
| V15 seed 0 (focus only), best sevenths | lag 8, natural prior | 81.98 | 77.73 | 84.82 | 81.51 |

## Per quality, all val sets pooled: recall / precision %

Recall: reference frames of the quality output as the same chord (root and quality). Precision: frames output as the quality that are that chord in the reference. dim includes dim7. Reference seconds: maj 9904, min 2253, 7 1066, maj7 442, min7 1006, dim 64, aug 23, hdim7 143.

| system | point | maj | min | 7 | maj7 | min7 | dim | aug | hdim7 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V14 n_seed0 | network | 82.9 / 79.1 | 65.7 / 66.1 | 27.2 / 42.9 | 52.6 / 51.1 | 56.0 / 60.0 | 0.0 / - | 0.0 / - | 0.0 / - |
| V14 n_seed0 | lag 8 | 83.9 / 80.5 | 69.2 / 67.9 | 31.2 / 41.6 | 60.0 / 51.3 | 58.3 / 58.9 | 0.0 / - | 0.0 / - | 0.0 / - |
| V15 seed 0 (62.5 % replay) | network | 83.0 / 78.8 | 63.5 / 68.5 | 34.1 / 47.8 | 54.8 / 50.8 | 54.3 / 62.2 | 6.0 / 14.1 | 18.8 / 13.2 | 32.4 / 65.3 |
| V15 seed 0 (62.5 % replay) | lag 8, mix prior | 83.9 / 80.0 | 67.4 / 70.6 | 36.1 / 46.8 | 60.9 / 49.9 | 57.4 / 62.0 | 2.5 / 6.0 | 25.2 / 21.6 | 33.7 / 68.2 |
| V15 seed 0 (62.5 % replay) | lag 8, natural prior | 83.4 / 80.4 | 67.5 / 71.1 | 38.1 / 46.0 | 61.7 / 49.1 | 57.8 / 62.0 | 6.0 / 10.6 | 34.4 / 16.5 | 39.3 / 58.3 |
| V15 seed 0 (62.5 % replay), best sevenths | network | 81.9 / 79.5 | 63.9 / 68.4 | 35.2 / 42.1 | 55.8 / 51.4 | 58.8 / 58.6 | 0.0 / - | 0.0 / - | 0.0 / - |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, mix prior | 82.6 / 80.4 | 67.2 / 70.3 | 38.2 / 41.8 | 63.5 / 50.4 | 61.2 / 57.7 | 0.0 / - | 0.0 / - | 0.0 / - |
| V15 seed 0 (62.5 % replay), best sevenths | lag 8, natural prior | 82.0 / 80.7 | 67.4 / 70.4 | 40.5 / 40.7 | 64.1 / 48.7 | 61.6 / 57.4 | 0.0 / - | 0.0 / - | 0.0 / - |
| V15 seed 0 (50 % replay) | network | 82.5 / 78.9 | 63.3 / 69.5 | 33.3 / 47.8 | 54.8 / 49.5 | 56.9 / 61.4 | 9.7 / 19.9 | 20.4 / 10.4 | 37.6 / 60.2 |
| V15 seed 0 (50 % replay) | lag 8, mix prior | 83.6 / 79.6 | 67.1 / 71.2 | 33.8 / 46.6 | 60.8 / 49.2 | 59.8 / 62.2 | 5.2 / 12.8 | 24.8 / 14.7 | 39.1 / 64.1 |
| V15 seed 0 (50 % replay) | lag 8, natural prior | 82.9 / 80.2 | 67.1 / 71.7 | 37.0 / 45.4 | 61.7 / 48.8 | 60.4 / 62.2 | 9.9 / 15.3 | 30.4 / 11.9 | 45.3 / 53.0 |
| V15 seed 0 (50 % replay), best sevenths | network | 81.0 / 79.9 | 63.6 / 69.4 | 39.3 / 42.9 | 56.7 / 47.9 | 60.7 / 59.9 | 3.6 / 15.8 | 8.4 / 11.7 | 29.1 / 71.5 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, mix prior | 81.8 / 80.6 | 67.4 / 72.1 | 40.7 / 41.4 | 62.3 / 46.8 | 64.9 / 60.3 | 0.9 / 4.7 | 8.8 / 29.3 | 29.4 / 80.4 |
| V15 seed 0 (50 % replay), best sevenths | lag 8, natural prior | 81.1 / 81.3 | 67.7 / 72.5 | 44.3 / 40.9 | 63.1 / 45.8 | 65.0 / 60.5 | 0.9 / 2.7 | 10.4 / 9.2 | 36.6 / 61.7 |
| V15 seed 0 (focus only) | network | 79.3 / 79.6 | 62.9 / 66.4 | 41.4 / 41.6 | 56.5 / 44.7 | 56.5 / 59.7 | 7.8 / 11.6 | 21.2 / 8.2 | 40.0 / 49.8 |
| V15 seed 0 (focus only) | lag 8, mix prior | 81.2 / 80.4 | 66.2 / 69.6 | 40.5 / 42.0 | 61.3 / 45.5 | 61.9 / 61.6 | 3.8 / 7.9 | 24.0 / 12.7 | 41.6 / 59.2 |
| V15 seed 0 (focus only) | lag 8, natural prior | 79.4 / 81.5 | 66.1 / 69.7 | 46.3 / 40.6 | 62.7 / 43.3 | 62.4 / 61.1 | 11.0 / 12.9 | 34.8 / 8.8 | 47.3 / 41.9 |
| V15 seed 0 (focus only), best sevenths | network | 82.9 / 79.1 | 65.7 / 66.1 | 27.2 / 42.9 | 52.6 / 51.1 | 56.0 / 60.0 | 0.0 / - | 0.0 / - | 0.0 / - |
| V15 seed 0 (focus only), best sevenths | lag 8, mix prior | 84.9 / 79.6 | 69.2 / 68.0 | 25.0 / 42.1 | 56.4 / 53.6 | 57.1 / 60.3 | 0.0 / - | 0.0 / - | 0.0 / - |
| V15 seed 0 (focus only), best sevenths | lag 8, natural prior | 83.9 / 80.5 | 69.2 / 67.9 | 31.2 / 41.6 | 60.0 / 51.3 | 58.3 / 58.9 | 0.0 / - | 0.0 / - | 0.0 / - |
