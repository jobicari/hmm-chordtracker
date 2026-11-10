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
| V8 | Sevenths: 60 classes (maj, min, 7, maj7, min7) and a curated ChoCo export (344 pop and jazz recordings, 18.9 h) added to training, on V7.3's causal model and decoders; on validation the network alone gains 8.9 points sevenths on RWC-P (68.5% vs 59.6%) at the same majmin, the decoder divides by the training prior to the power 0.25 (the full prior floods the output with 7ths), maj6, sus4 and dominant 9 are not learned | 78.0 % (network alone, 0.28 s; RWC-P 80.2 %, GuitarSet 75.9 %); sevenths 62.1 %; ChoCo 83.4 % / sevenths 74.8 % | 79.4 % (fixed-lag Viterbi, 0.65 s; RWC-P 81.4 %, GuitarSet 77.4 %); sevenths 63.8 %; ChoCo 84.4 % / sevenths 76.1 % | [reports/v8.md](reports/v8.md) |
| V9 | Class-weighted cross-entropy on V8 (weight = train share of the quality ^ -alpha, mean 1, alpha 0.25 and 0.5); null result on validation: 7 recall rises (ChoCo network alone 21.0% to 55.2% at alpha 0.5) but maj recall falls more, and sevenths drops on every set (best mean 71.06% and 70.24% vs V8's 72.99%), so V9 keeps V8 | 78.0 % (V8, not rerun) | 79.4 % (V8, not rerun) | [reports/v9.md](reports/v9.md) |
| V10 | Seven small changes around V8, each alone: an ensemble of V8's two seeds, per-quality posterior offsets, a tuning estimate that does not wrap at +-25 cents (issue #11), a no-chord gate on the input level, early stopping on sevenths, root / quality / bass heads, and gain, tilt and noise augmentation; mostly null on validation: only the ensemble gains (73.32% vs 72.99% mean sevenths at 1.02 s, about half the seed gap, at the cost of two networks), the retrains stay within seed noise or lose (augmentation -1.3), so V10 keeps V8 and the tuning fix | 78.0 % (V8, not rerun) | 79.4 % (V8, not rerun) | [reports/v10.md](reports/v10.md) |
| V11 | Distilling V8's two-seed ensemble into one causal network of V8's size (loss = CE on the labels + KL to the ensemble's posteriors); on the labelled audio alone a null result, but pre-trained first on 76 h of unlabelled music (KL only) the network alone gains on both seeds on validation (+0.76 and +0.71 mean sevenths over V8 with the same seed, lag 8 +0.57 and +0.16) | 78.1 % (network alone, 0.28 s; RWC-P 81.2 %, GuitarSet 75.0 %); sevenths 63.6 %; ChoCo 84.3 % / sevenths 75.5 % | 79.1 % (fixed-lag Viterbi, 0.65 s; RWC-P 82.0 %, GuitarSet 76.2 %); sevenths 64.8 %; ChoCo 85.4 % / sevenths 76.6 % | [reports/v11.md](reports/v11.md) |
| V12 | Three checks on V11's frozen network, no retraining: per-quality posterior offsets (null on validation again: fitted on one seed or on two sets they do not carry to the other seed or the third set, lag-8 sevenths -0.62 to -4.87 on the held-out set); the tuning estimate (the live chain uses V10's fix, but V11's RWC-P and GuitarSet training cache predates it: all 200 songs match the old estimate, 26 move by more than 3 cents, median 0.5); the listening prototype on V11's network with sevenths, steadiest by ear at 1.02 s | 78.1 % (V11, not rerun) | 79.1 % (V11, not rerun) | [reports/v12.md](reports/v12.md) |
| V13 | Beats, downbeats and the bass around V11's network, null result: beats in the decoder add nothing, annotated downbeats +0.85 mean sevenths at lag 8 on RWC-P and GuitarSet validation but madmom's (offline) only +0.39 and not across sets; about 1 to 2 of the 6.9 extra chord changes a minute follow the played bass (RWC-P's MIDI), and neither a bass head at decode time nor a played-bass loss in training removes them (-0.33 at lag 8); the cache rebuilt with the 12-bin tuning ties V11 (73.09 to 73.25 % vs 73.16 %), so V11 stays frozen | 78.1 % (V11, not rerun) | 79.1 % (V11, not rerun) | [reports/v13.md](reports/v13.md) |
| V14 | An N (no chord) class in V8's sevenths network (61 classes, trained on RWC-P and ChoCo's N), and a variant that also feeds the GRU the frame's input level in dBFS; null result on validation: the network finds more reference N than V10's level gate but also calls N on chords (29 to 53 s of GuitarSet, which has none), and at lag 8 its mean sevenths with N scored is 72.55 to 73.07 % (best decoder setting 73.44 %) against 73.97 % for V11 + gate; the level input does not help, so V11 stays frozen | 78.1 % (V11, not rerun) | 79.1 % (V11, not rerun) | [reports/v14.md](reports/v14.md) |
| V15 | Diminished, augmented and half-diminished classes added to V14's N network (97 classes, dim7 trained as dim), fine-tuned from it on excerpts around those chords (12 shifts, tempo +-6 %, gain +-3 dB) mixed with the normal training songs; on validation, with 62.5 % replay, hdim7 33.7 % recall (68.2 % precision) and aug 25.2 % (21.6 %) while mean sevenths at lag 8 stays at 73.34 % (V14 73.44 %) and tetrads rises to 66.71 % (66.32 %); dim is not learned (2.5 %) and the losses fall on GuitarSet; one seed, test not run | 78.1 % (V11, not rerun) | 79.1 % (V11, not rerun) | [reports/v15.md](reports/v15.md) |

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
| `src/chordtracker/data_choco.py` | V8: the curated ChoCo export (referenced, not copied) as songs; manifest tuning from the first frame; SHA-256 check of each WAV |
| `src/chordtracker/data_v8.py`, `model_v8.py`, `train_v8.py`, `transitions_v8.py` | V8: vocabularies A / B / B9 as targets; the n-class causal CRNN and its training; the prior, the tempered prior and the transition matrices; V9: the class weights of the loss (train.class_weight_alpha, default 0) |
| `configs/v8.yaml` | V8: export path, V7.2's model and decoders, prior power, the points frozen on val |
| `scripts/run_v8.py`, `scripts/v8_test.py` | V8: train + decode on val (--vocab, --seed, --transitions, --prior-power); test once over the frozen points |
| `scripts/v8_choco_split.py` | V8: the grouped ChoCo split |
| `src/chordtracker/frame_score.py` | V10: per-song scoring tables (matched and scored seconds per class and frame), equal to evaluate.score, so a decoded path is scored by lookup |
| `src/chordtracker/decode_v10.py`, `model_v10.py`, `train_v10.py` | V10: the decoder points on any posterior, quality offsets and the level gate; the CRNN with root / quality / bass heads; training with sevenths early stopping, the heads loss and input augmentation |
| `configs/v10.yaml`, `scripts/run_v10.py`, `scripts/train_v10.py` | V10: V8's settings plus the training options (off by default); the validation steps (cache, ensemble, offsets, retune, gate, points); the retrains |
| `scripts/v10_offsets.py`, `v10_gate.py`, `v10_tuning.py` | V10: the offsets with their two guards; the level gate; the tuning estimates before and after the fix |
| `src/chordtracker/train_v11.py` | V11: the ensemble's geometric-mean posterior as a soft target, the KL term, V8's training with it, and pre-training on unlabelled songs (KL only) |
| `configs/v11.yaml`, `scripts/train_v11.py`, `scripts/v11_pool.py` | V11: V8's settings plus kl_weight, the teachers and the points frozen on val; training (--pool for stage A first); the unlabelled pool's list, exclusions and feature cache |
| `scripts/run_v12.py`, `scripts/v12_tuning.py` | V12: V10's offsets and both guards on V11's networks; which tuning estimate each cached training song was built with |
| `src/chordtracker/midi.py`, `model_v13.py`, `train_v13.py` | V13: a standard-library MIDI reader (RWC-P's aligned MIDI, the bass tracks); the CRNN with a played-bass head; V11's stage B with the played-bass loss |
| `configs/v13.yaml`, `scripts/v12_beats.py`, `v13_madmom_downbeats.py`, `v13_bass.py`, `v13_cache.py`, `train_v13.py`, `v13_eval.py` | V13: beat-dependent p_self on validation (annotated, librosa and madmom beats); madmom's downbeats (separate environment); the bass measure and bass heads at decode time; the cache with the 12-bin tuning; the retrains; their validation |
| `src/chordtracker/features_v14.py`, `model_v14.py`, `train_v14.py` | V14: the frame's input level in dBFS from past audio (cached and live, the same function); the CRNN with an N class and optionally the level as a GRU input, offline and one column at a time; V8's training with N kept under the pitch shift |
| `configs/v14.yaml`, `scripts/v14_census.py`, `scripts/run_v14.py` | V14: V8's settings with N; where the reference N lies; training, the V8 baselines on the same features, and the validation scores with N, N-excluded and N seconds |
| `src/chordtracker/data_v15.py`, `train_v15.py`, `eval_v15.py` | V15: vocabulary A+ (V14's classes plus dim, aug and hdim7 after N, and the decode order with N last); extending V14's output layer, focus excerpts with a tempo stretch on the cached log-CQT, replay batches and the epoch criterion; tetrads, mirex and per-quality recall and precision |
| `configs/v15.yaml`, `scripts/v15_census.py`, `scripts/run_v15.py` | V15: the fine-tune settings; how much dim, aug, dim7 and hdim7 the sets hold; validation tables, the fine-tunes and the validation scores against V14 |
| `src/chordtracker/play_along.py`, `scripts/play_along.py` | Listening prototype (V15's network by default, `--model v15`; V11's and V14's selectable): what to show when, the live analysis in its own process; the player (input device or file, delayed playback, window, keys, log, `--dry-run` timing) |
| `src/chordtracker/demo.py`, `scripts/demo_export.py`, `scripts/demo_server.py`, `demo/`, `Dockerfile` | Demo website (V15's network by default; V14's and V7.2's selectable): outputs of GuitarSet validation takes exported with the prototype's chain; a page that plays a recording and shows its chords live, decoded on the demo server as it plays, framewise vs fixed-lag Viterbi; packaged as a Docker image ([reports/demo.md](reports/demo.md), [demo/README.md](demo/README.md)) |
| `src/chordtracker/inference.py`, `src/chordtracker/weights/` | The frozen V6 (bidirectional, seed 1) as a library: whole-song posteriors of an audio file, its settings and weights packaged (see [Use from another project](#use-from-another-project)) |
| `splits/guitarset.json` | the fixed song-level split (seed 0), tracked |
| `splits/rwc_p.json` | RWC-P 60 / 20 / 20 split by artist |
| `splits/choco.json` | ChoCo export 237 / 47 / 60 recordings by release (isophonics, robbie-williams) or artist (uspop2002), seed 0; jaah train only |
| `data/` | GuitarSet, RWC-P and everything generated from them (git-ignored) |
| `results/<version>/<split>/` | `summary.json` and `per_song.csv` of each run |
| `figures/<version>/val/` | the saved report figures (val only) |
| `reports/<version>.md` | each version's write-up: question, setup, val analysis, test score, findings |

## Use from another project

The frozen V6 ships with the package: install it from git and read a recording's chord posteriors.

```python
from chordtracker import inference

times, post = inference.posteriors("song.wav", phases=4)    # (n,), (24, n) in inference.CLASSES order
```

`phases=4` reads the song four times, a quarter hop apart, for a frame every 23 ms instead of 93 ms
(finer sampling, not finer timing). The model runs on the CPU and is loaded once per process.

The tuning is estimated as V6's features do, on quarter-tone bins: it wraps at +-25 cents, so a recording
30-50 cents off A440 reads its chords a semitone off. A caller that knows the tuning passes it,
`tuning_semitones=-0.44` for a recording 44 cents flat.

## Data and licence

This project is for research and non-commercial use only, and so are its trained weights.

- GuitarSet: CC BY 4.0.
- RWC Popular Music (RWC-P): CC BY-NC 4.0; audio not included.
- ChoCo (Chord Corpus, smashub/choco): chord labels CC BY 4.0, except the JAAH partition, CC BY-NC-SA 4.0.
  V8 trains on a curated export of its Isophonics, USPOP2002, Robbie Williams and JAAH partitions
  (choco_corpus_prep, commit 896a9f8), referenced through `data.choco_export` and
  `data/choco/SOURCE.txt`. The recordings are copyrighted and are not included or redistributed.

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
