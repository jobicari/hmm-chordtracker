RWC-P validation, 19 songs with a MIDI bass part (77.7 min); left out: rwc/RWC_P078.
V11 seed 1, fixed-lag Viterbi lag 8, p_self 0.98. Held chord: annotation run of one vocabulary-A chord, bass ignored; inside: more than 0.3 s from its ends; with the bass: a MIDI bass change within 0.15 s.

| quantity | count | per minute |
|---|---:|---:|
| reference chord changes (vocabulary A, bass ignored) | 2147 | 27.6 |
| decoded chord changes | 2681 | 34.5 |
| decoded changes inside a held chord | 490 | 6.3 |
| of them, with a bass change within 0.15 s | 251 | 3.2 |
| of those, the new chord's root is the new bass note | 165 | 2.1 |
| MIDI bass changes inside held chords | 4144 | 53.3 |

Share of the changes inside a held chord with a bass change near: 51.2 %; chance (share of the time inside held chords within 0.15 s of a bass change): 33.0 %.
Bass changes inside held chords followed by a decoded change: 6.1 %.
Alignment check: the MIDI bass note's pitch class equals the annotated bass on 78.7 % of the annotated time (song mean).
