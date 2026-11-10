# V15 census: dim, aug, dim7, hdim7

Seconds from the annotations, train and val only. Groups as scripts/v15_census.py reads the labels (bass ignored, extensions above the octave dropped). Cells: seconds / segments / songs.

## Context

| set | split | songs | labelled s | A + N trained s |
|---|---|---:|---:|---:|
| RWC-P | train | 60 | 14420 | 13340 |
| RWC-P | val | 20 | 4954 | 4459 |
| GuitarSet | train | 60 | 1663 | 1663 |
| GuitarSet | val | 60 | 1945 | 1852 |
| ChoCo | train | 237 | 48288 | 44895 |
| ChoCo | val | 47 | 9529 | 8685 |
| all | train | 357 | 64371 | 59899 |
| all | val | 127 | 16427 | 14997 |

## train

| group | RWC-P | GuitarSet | ChoCo | all s | share of A + N trained | inversion s |
|---|---:|---:|---:|---:|---:|---:|
| dim | 130 / 94 / 26 | - | 139 / 97 / 23 | 269 | 0.45 % | 25 |
| aug | 85 / 53 / 13 | - | 193 / 115 / 24 | 278 | 0.46 % | 14 |
| dim7 | 15 / 8 / 5 | - | 121 / 123 / 16 | 136 | 0.23 % | 10 |
| hdim7 | 50 / 32 / 11 | - | 81 / 52 / 12 | 131 | 0.22 % | 21 |
| min7(b5) | - | - | 12 / 8 / 1 | 12 | 0.02 % | 0 |
| aug(b7) | 1 / 1 / 1 | - | 11 / 13 / 2 | 13 | 0.02 % | 0 |
| maj(#5) | - | - | 1 / 2 / 2 | 1 | 0.00 % | 1 |

## val

| group | RWC-P | GuitarSet | ChoCo | all s | share of A + N trained | inversion s |
|---|---:|---:|---:|---:|---:|---:|
| dim | 15 / 9 / 3 | - | 49 / 15 / 4 | 63 | 0.42 % | 13 |
| aug | 15 / 8 / 2 | - | 8 / 5 / 1 | 23 | 0.15 % | 0 |
| dim7 | - | - | 1 / 1 / 1 | 1 | 0.00 % | 1 |
| hdim7 | 35 / 13 / 2 | 92 / 48 / 24 | 14 / 7 / 2 | 141 | 0.94 % | 6 |
| aug(b7) | 1 / 1 / 1 | - | - | 1 | 0.00 % | 0 |

## Spellings (train and val, all sets)

| group | most common spellings (count of segments) |
|---|---|
| dim | dim (192), dim/b3 (15), dim/b5 (8) |
| aug | aug (171), aug/3 (7), aug/#5 (2), aug(9,11) (1) |
| dim7 | dim7 (123), dim7/b5 (5), dim7/b3 (4) |
| hdim7 | hdim7 (133), hdim7/b7 (11), hdim7/b5 (4), hdim7/b3 (3), dim/b7 (1) |
| min7(b5) | min7(b5) (8) |
| aug(b7) | (3,#5,b7,b9) (11), aug(b7) (4) |
| maj(#5) | A/b6 (1), E/b6 (1) |
