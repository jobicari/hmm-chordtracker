# Training and decoding with the bass note: plan

Planning only; no code or training yet. The listening finding it answers: a moving bass under a held
chord (walking lines, passing notes, inversions) often pulls the chord off.

## What the labels carry

Harte labels can carry the bass (C:maj/3 = C major with E in the bass). Share of the annotated time
with a bass other than the root, training and validation (test not read):

| set | split | songs | labels with a bass | time with bass not the root | of it, in vocabulary A | bass-only moves / min |
|---|---|---:|---:|---:|---:|---:|
| RWC-P | train | 60 | 1,003 of 8,174 | 10.7 % | 8.7 % | 0.20 |
| RWC-P | val | 20 | 300 of 2,595 | 10.2 % | 8.1 % | 0.16 |
| GuitarSet | train | 60 | 0 of 480 | 0 % | 0 % | 0 |
| GuitarSet | val | 60 | 0 of 912 | 0 % | 0 % | 0 |
| ChoCo, Beatles validation albums | val | 25 | 278 of 2,158 | 8.0 % | 6.9 % | 0.49 |

"Bass-only moves" counts consecutive segments with the same root and quality and a different bass
(C -> C/3). ChoCo's other partitions (uspop2002, Robbie Williams, JAAH) and its training split are in
the export outside this folder and were not counted.

Two consequences:

- **GuitarSet has no bass in its labels** (its chords are lead-sheet chords), and RWC-P and the
  Beatles mark an inversion on about a tenth of the time.
- **Walking and passing bass notes are not annotated at all.** An annotator writes C for a bar whose
  bass walks C-D-E-F; at most 0.2 to 0.5 held inversions a minute appear as labels. So the labels can
  teach the network *held* inversions, not that a passing bass note leaves the chord unchanged. For
  that, the actual bass notes are needed: RWC-P has aligned MIDI for all 100 songs
  (`data/rwc/rwc-annotations/01_annotations_preprocessed/MIDI_aligned/`), with the bass on its own
  track. No MIDI reader is installed; a small standard-library reader would do.

## What V10 step 6 did

Three auxiliary linear heads on the GRU state next to the 60-class head: root (12), quality (5) and
bass (12, the lowest note of the label). Loss: chord CE + 0.5 x (root + quality + bass CE). Decoding
read the chord head only, and the scores (majmin, sevenths) ignore the bass. Result, one seed: RWC-P
+1.27 sevenths at lag 8, ChoCo -1.04, mean +0.07, within seed noise. The bass head was trained but
never used or scored. The model is still on disk (`data/v10_results/heads_seed0/model.pt`).

## Two options

**(i) A bass head used at decode time.** The network predicts chord and bass separately; the decoder
tracks both, so a change of bass alone is a bass event, not a chord event.
- Decoder: a joint state (chord, bass) is 60 x 12 = 720 states, too many for the fixed-lag Viterbi at
  this frame rate. Cheaper: two coupled chains, the chord chain as today, plus a bass chain whose
  changes are free while the chord holds; or a chord transition that is made dearer when the bass head
  alone moves (the evidence that changed is bass evidence).
- Output: slash chords (C/E) in the prototype when the bass head is confident and differs from the root.
- Training: V10's heads recipe, the bass head possibly weighted up. A first look needs no training:
  decode V10's heads model with its bass head, on validation.

**(ii) Bass-aware targets.**
- Chord x bass classes: 60 chords x 3 bass positions (root, third, fifth; the seventh for tetrads)
  = 180 to 240 classes. The network learns that C/E is still C, as one class family. Rare classes are
  thin: 8-10 % of RWC-P time, none of GuitarSet.
- Or chord + bass in the loss with the bass target from the *played* bass (RWC-P's MIDI), not the
  label: the only source that has the walking lines. That is the version that addresses the
  listening finding directly, but only RWC-P has it (about 4 h of training audio).

## How a gain would show

- **mir_eval's inversion-aware metrics**, `majmin_inv` and `sevenths_inv` (available in mir_eval
  0.8.2): a frame labelled C:maj/3 counts only if the root, quality and bass all match. Both stay
  reported next to majmin / sevenths, which must not drop.
- **Chord changes during a held chord**: in each reference segment, the decoded chord changes that
  fall inside it, per minute. Split by whether the bass moved there (RWC-P's MIDI bass track): the
  "changes caused by bass-only movement" count. Today's lag 8 makes 32 to 35 changes a minute against
  28 in the reference; the excess is where to look.
- **By ear**, on the songs with busy bass lines.

## Order

1. Measure first, no training: on RWC-P validation, the chord changes inside held reference segments
   at lag 8, and how many coincide with a bass-note change in the MIDI. If few do, the bass is not the
   main cause and the plan stops here.
2. Decode-only: V10's heads model with its bass head (option i), scored with the _inv metrics and the
   change count.
3. Then a retrain (option i weighted, or option ii), which is also when the RWC-P and GuitarSet
   feature cache is rebuilt with the 12-bin tuning estimate (`data/processed/v7_2_logcqt_Fs1` moved
   aside, not deleted, and recomputed).
