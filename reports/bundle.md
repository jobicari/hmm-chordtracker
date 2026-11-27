# Bundle: the frozen V15 for a port of the live chain

## Question

The live chain (`play_along.Analysis`: block-by-block causal log-CQT, the causal CRNN, fixed-lag Viterbi)
runs in Python on numpy, librosa and torch. A standalone macOS app written in Swift cannot import it.
**What does a port need, so that it outputs the same chords, frame for frame, as the demo website does
with V15?** This file answers that. It is also the bundle's README: `scripts/export_bundle.py` copies it
into every bundle.

## Summary of the results

`scripts/export_bundle.py` writes the frozen V15 (results/v15/ft_seed0, the demo's default system) as a
bundle of 15 MB:

1. **The network runs in Core ML with the same results.** The CRNN's step for one frame
   (`ChordStep.mlpackage`, float32) gives posteriors within 7.5e-7 of PyTorch's, and the decoded labels are
   identical at all four points on both golden recordings: 383 of 383 frames (funk) and 275 of 275 frames
   (jazz_flat_silence) at 0.28, 0.46, 0.65 and 1.02 s.
2. **The decoder is two files per point.** The prior the evidence divides by and each point's transition
   matrix are exported as numbers, so the port needs neither the configs nor the Billboard counts.
3. **Every stage has golden numbers.** Two GuitarSet validation takes run through the live chain with every
   intermediate result kept: input, harmonic part, frame levels, the tuning estimate and what it is
   computed from, |CQT|, log-CQT, posteriors and the labels of every point. A port checks each of its
   stages against them in turn.
4. **The export environment does not change the chords.** It runs torch 2.7.0 (coremltools 9.0 converts
   torch up to 2.7.0), the project's environment torch 2.14.1. Recomputed in the project's environment,
   the golden harmonic part, |CQT|, log-CQT and posteriors are bitwise the same and every label is equal.

## What the bundle holds

    v15/
      README.md               this file
      system.json             the settings of every stage, the 97 labels and their names, the four points
                              with their delays, and the decoder arrays' descriptions
      decoder/prior.f64       (97,) the decoding prior
      decoder/lag2.f64        (97, 97) transition matrix A of the 0.46 s point (lag 2, p_self 0.5)
      decoder/lag4.f64        the 0.65 s point (lag 4, p_self 0.8)
      decoder/lag8.f64        the 1.02 s point (lag 8, p_self 0.98)
      ChordStep.mlpackage     the network, one frame per call
      golden/funk/                00_Funk2-108-Eb_comp, the demo's take, 35.5 s, 8 cents sharp
      golden/jazz_flat_silence/   02_Jazz2-110-Bb_comp's first 24 s shifted 35 cents flat, after 1.5 s of
                                  digital silence (the estimate reads 30 cents flat)

**Arrays** are raw little-endian files in row-major order (C order), with no header. Each is described
where it is used by `{"file", "dtype", "shape"}`: dtype `f32` (float32), `f64` (float64), `i32` (int32)
or `c64` (complex64, real then imaginary float32). Example: `{"file": "decoder/lag4.f64", "dtype": "f64",
"shape": [97, 97]}` is 75,272 bytes, `A[i, j]` at byte offset `8 * (97 * i + j)`.

**Labels** are decode indices 0 to 96: 12 roots (C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B) of maj, min, 7,
maj7, min7, dim, aug, hdim7 in that order, then N (96). `system.json` gives each its Harte label and the
name a musician reads (`C:min7` is `Cm7`, `B:hdim7` is `Bm7b5`, `N` is `N.C.`).

**A golden case** is a folder with `manifest.json` and its arrays. With T frames and n samples:

| Array | dtype, shape | What it is |
|---|---|---|
| `input` | f32 (n,) | mono audio at 22,050 Hz, the chain's input |
| `harmonic` | f32 (n,) | the harmonic part (stage 1) |
| `levels` | f64 (T,) | dBFS of every frame's own samples (stage 2); `-inf` is digital silence |
| `tuningHarmonic` | f32 (220500,) | the harmonic part of the 10 s the tuning is estimated on (stage 3) |
| `tuningPitches` | f32 (m,) | the peak frequencies, Hz, of the tuning histogram (stage 3) |
| `cqt` | f64 (T, 168) | \|CQT\| of every column, before the level and the log (stage 4) |
| `kernelA440Bin0`, `...Bin84`, `...Bin167` | c64 | the trimmed A440 kernels of three bins (stage 4) |
| `logcqt` | f32 (T, 168) | the columns the network reads (stage 5) |
| `posteriors` | f32 (T, 97) | P(class \| frame), decode order (stage 6) |
| `labels/lag0`, `lag2`, `lag4`, `lag8` | i32 (T,) | the label of every frame at each point (stage 7) |

`manifest.json` adds the tuning (`startSample`, `bins`, `semitones`, `firstTunedColumn`) and a table of
the 168 CQT bins at A440 and at the estimated tuning (`hz`, `length`, `taps`, `firstTap`, `delayFrames`,
`scale`). Example, funk at A440, bin 0 (F#1): 46.249 Hz, length 16,512.39, 16,512 taps from 8,255
samples before its centre, delayed 5 frames, scale 128.50.

## The live chain, stage by stage

Symbols: sr = 22,050 Hz; hop h = 2,048 samples (92.88 ms); frame f is the input samples
`f h .. (f + 1) h`; T = 1 + n // h frames for n samples. Every stage keeps only what the past requires.
The numbers come from `system.json`; the Python functions named are the reference (hmm-chordtracker,
`src/chordtracker/`), librosa 1.0.0.

**0. Input.** Mono float32 at 22,050 Hz: the mean of the channels, resampled from the device's rate. The
golden cases start here, so the resampler is not checked by them; use a high-quality one.

**1. Harmonic part** (`streaming_features.StreamingFeatures`, STFT frames of 2,048 every 512 samples):

- STFT frame j reads input samples `512 j - 1024 .. 512 j + 1024`, zeros outside the recording, times the
  periodic Hann window `w[k] = 0.5 - 0.5 cos(2 pi k / 2048)` (float32); `S = rfft` (1,025 bins), `M = |S|`.
- Time median: `H[b] = median of M[b]` over frames `j - 8 .. j` (9 values; frames before the recording
  count as zeros).
- Frequency median: `P[b] = median of M[b - 15 .. b + 15]` of the same frame, mirrored at the edges
  (scipy's `reflect`: index -1 reads bin 0, -2 reads bin 1, 1025 reads bin 1024).
- Soft mask (librosa.util.softmask, power 2): `Z = max(H, 4 P)`; where `Z` is below the smallest normal
  float32 the mask is 0, else `mask = (H / Z)^2 / ((H / Z)^2 + (4 P / Z)^2)`.
- `y_j = w * irfft(S * mask, n = 2048)`, overlap-added at samples `512 j - 1024 ..`, and the squared
  window summed at the same places. Harmonic sample i = sum / squared-window sum (the sum itself where
  that is below the smallest normal float32).
- Sample i is final once no later frame reaches it: after frame j, samples up to `512 (j + 1) - 1024`.
  Frame j is computed once the input reaches `512 j + 1024`. At the end of a recording of n samples,
  frames continue up to `j = n // 512` with zeros after n, and all n samples are final.

**2. Levels** (`streaming_features.dbfs`): frame f's level is `10 log10(mean(x^2))` of its own input samples
(float64), `-inf` for silence. The tuning seconds start at the first frame whose level is at or above
-55 dBFS (`tuning.audibleDbfs`): `startSample = f h`. Example: jazz_flat_silence has 1.5 s of zeros, so
frames 0 to 15 are `-inf` and the start is frame 16, sample 32,768.

**3. Tuning** (`features_v7_1.start_tuning`): A440 until 10 s (220,500 samples) of input have arrived from
the start, then, once the input is longer than `startSample + 220500`:

- The harmonic part of those 220,500 samples alone (stage 1 on that segment, finished as a recording of
  220,500 samples): `tuningHarmonic`. The reference computes it with librosa's stft and istft; stage 1 run
  on the segment gives it within 1.1e-7 and the same tuning on both golden cases.
- Peaks (librosa.piptrack, the defaults of librosa.estimate_tuning): the centred STFT of it (2,048, hop
  512, zeros padded 1,024 at both ends, periodic Hann), `S = |STFT|`. In every frame, bin k is a peak if
  its frequency `k sr / 2048` is in [150, 4000) Hz and it is a local maximum of `S * (S > 0.1 max(S of
  the frame))`: `x[k] > x[k - 1]` and `x[k] >= x[k + 1]`, never bin 0, the last bin if `x[K-1] > x[K-2]`.
  Its pitch is `(k + s) sr / 2048` with `s = -b / a`, `a = S[k+1] + S[k-1] - 2 S[k]`,
  `b = (S[k+1] - S[k-1]) / 2` (`s = 0` if `|b| >= |a|`), and its magnitude `S[k] + 0.5 g s`, g the
  gradient of S along frequency (central difference).
- `tuningPitches`: the peaks whose magnitude is at least the median magnitude of all peaks.
- Histogram (librosa.pitch_tuning, 12 bins per octave, resolution 0.01): residual
  `r = (12 log2(f / 27.5)) mod 1`, minus 1 where `r >= 0.5`; 100 bins from -0.5 to 0.5 (numpy.histogram,
  last bin closed); the tuning in semitones is the left edge of the fullest bin (the first one on ties).
  Example: funk +0.08 (8 cents sharp), jazz_flat_silence -0.30.
- In CQT bins (24 per octave): `bins = 2 x semitones`. The tuned kernels apply from column
  `firstTunedColumn = ceil((startSample + 220500) / h) - 1` on (funk 107, jazz_flat_silence 123).

**4. CQT** (`streaming_features.Kernels`), 168 bins from F#1, 24 per octave, at a tuning of `u` bins:

- Frequency `f_k = 46.2493 x 2^(u / 24) x 2^(k / 24)` Hz. Length `L_k = Q sr / f_k`, `Q = 1 / alpha`,
  `alpha = (2^(1/12) - 1) / (2^(1/12) + 1)` (Q = 34.634).
- Kernel (librosa.filters.wavelet, norm 1): `N_k = floor(L_k / 2) - floor(-L_k / 2)` samples;
  `z[i] = exp(2 pi i f_k / sr x (i + floor(-L_k / 2))) x hann_periodic(N_k)[i]`, divided by `sum |z|`,
  rounded to complex64. Sample 0 is zero (the window), so the kernel's taps are `z[1 ..]`, `N_k - 1` of
  them. With `W = ceil(max_k L_k)`, the first tap sits `firstTap_k = (W - N_k) // 2 + 1 - W // 2` samples
  from the kernel's centre. The kernels' phase reference does not matter (only |CQT| is used).
- Delay `d_k = ceil(L_k / 2 / h)` frames at A440; the tuned kernels are delayed by the larger of their
  own and A440's, so every kernel ends at its frame.
- Column t, bin k: `C[t, k] = |sum_m tap_k[m] y[(t - d_k) h + firstTap_k + m]| x sqrt(L_k)`, y the
  harmonic part, zeros outside the recording, summed in double precision.
- Column t is computed once the harmonic part is final up to `t h + max_k(firstTap_k + taps_k - d_k h)`
  and, from `firstTunedColumn` on, once the tuning is known. At the end of a recording, every column up
  to T - 1 is computed with zeros after it.

**5. Level and log** (`streaming_features.sorted_percentile`): every |CQT| value of columns 0 to t (168
per column, float64) in one sorted list; `level_t = max(v[lo] + (v[lo + 1] - v[lo]) (p - lo), 1e-10)` with
`p = 0.99 (N - 1)`, `lo = floor(p)` (numpy's linear percentile 99); `logcqt[t, k] = log1p(10 C[t, k] /
level_t)`, float32.

**6. Network** (`ChordStep.mlpackage`, `bundle.FrameStep`): rows 12 to 155 of each column (C2 to B7, 144
values) go into a window of 15 columns, zeros at first, oldest column first. When column t arrives the
window moves by one; once more than 2 columns have arrived, one call `(window, hidden) -> (posteriors,
hidden_out)` gives P(class | frame t - 2), and `hidden_out` is the next call's `hidden` (zeros before the
first call). At the end of a recording, 2 zero columns are added, for the last two frames. Shapes: window
(1, 1, 144, 15), hidden (1, 1, 64), posteriors (97,). Core ML on the CPU in float32 (`.cpuOnly`) gives
the golden posteriors; other compute units were not checked.

**7. Decoders** (`online_hmm`), one per point, on each frame's posteriors p:

- 0.28 s (lag 0): the network alone, `argmax p` (the first maximum on ties).
- Fixed-lag Viterbi (lag L, matrix A of the point, prior q, eps = 1e-10): evidence
  `e[j] = log(p[j] + eps) - log(q[j] + eps)`, `logA = log(A + eps)`, float64. First frame:
  `delta = e - log(97)`. Each next frame: `s[i, j] = delta[i] + logA[i, j]`, `psi[j] = argmax_i s[i, j]`
  (first on ties), `delta[j] = s[psi[j], j] + e[j]`, then `delta -= max(delta)`; keep the last L rows of
  psi. From frame L on, each frame t decides frame t - L: start at `argmax delta`, then follow
  `psi_t, psi_{t-1}, ..., psi_{t-L+1}`. At the end of a recording, the last L frames are read off the final
  best path (start at `argmax delta`, follow the kept psi rows backwards).

## Points, delays and what is shown when

As on the demo page (`chordtracker.demo`, `demo/app.js`). Every point in `system.json` has:

| Point | Decoder | Look-ahead (frames) | Decision delay D | Playback delay P = D + h / 2 |
|---|---|---|---|---|
| 0.28 s | network alone | 3 = 2 input + 1 harmonic | 0.372 s (4 h) | 0.418 s |
| 0.46 s | lag 2, p_self 0.5 | 5 | 0.557 s | 0.604 s |
| 0.65 s (default) | lag 4, p_self 0.8 | 7 | 0.743 s | 0.789 s |
| 1.02 s | lag 8, p_self 0.98 | 11 | 1.115 s | 1.161 s |

- **Decision delay** `D = (lookaheadFrames + 1) h`: frame f's label is known once the input has reached
  `f h + D` (one frame of margin over the look-ahead), and a fixed-lag decision never changes afterwards.
- **Segments**: a run of equal labels is one chord; frame f's segment runs from `(f - 1/2) h` to
  `(f + 1/2) h` (the first from 0, the last to the end of the recording), as every WCSR in the project
  cuts them.
- **Delay on** (the default): playback is held back by `P + lead` (+ a margin for the app's own processing
  and audio latency), so each chord is decided before its segment is heard; the half frame in P is there
  because a segment starts half a frame before its frame. With a lead of 0.1 to 0.5 s, the chord shown is
  the one of the moment `lead` seconds ahead of what is heard, and the delay grows by as much.
- **Delay off**: the audio is not held back, and each chord appears D after its moment.
- The page shows the chosen point's labels and, below them, the network alone (0.28 s), whose flicker
  the decoder removes. The demo server adds 0.15 s of margin for the network and its round trip; a local
  app measures its own.

## Checks

| Check | Result |
|---|---|
| Core ML vs PyTorch, posteriors (max abs difference over all frames) | funk 7.45e-7, jazz_flat_silence 5.36e-7 |
| Core ML posteriors decoded with the bundle's prior and matrices vs the golden labels | 100 % of frames at all four points, both cases (383 and 275 frames) |
| Streamed log-CQT vs recomputed from the golden \|CQT\| (stage 5 on `cqt`) | max abs difference 1.2e-7 and 3.9e-7 |
| Streamed harmonic part vs librosa's on the whole recording (`features_v7_1.causal_harmonic`) | max abs difference 2.2e-8 and 8.9e-8 (signal RMS 0.014 and 0.047) |
| The golden cases in the project's environment (torch 2.14.1) vs the bundle's (torch 2.7.0) | harmonic part, \|CQT\|, log-CQT and posteriors bitwise equal; every label equal |
| `tests/test_bundle.py` | arrays read back as written; the replayed step equals the streaming CRNN; the points, delays and matrices of system.json; a capture has the live chain's labels and every stage; a bundle checks itself |

## What a port must reproduce

On both golden cases, fed `input` in blocks of any size and then finished:

- the tuning in the same histogram bin (`semitones` equal) and the same `startSample` and
  `firstTunedColumn`;
- `harmonic`, `cqt`, `logcqt` and `posteriors` to float32 precision: Python's own two computations differ
  by up to 9e-8 (harmonic part) and 4e-7 (log-CQT), Core ML and PyTorch by 7.5e-7 (posteriors);
- every label of every point.

A difference in a label means some stage drifted: the stage-by-stage arrays locate it.

## Reproduce

```bash
python3.12 -m venv .venv-export && .venv-export/bin/pip install -r requirements-export.txt
PYTHONPATH=src .venv-export/bin/python scripts/export_bundle.py --out <folder>     # writes <folder>/v15/
PYTHONPATH=src python -m pytest tests/test_bundle.py
```

The export reads GuitarSet from `data/guitarset/` and ends with the Core ML check, which compiles the
model in the user's temporary folder: run it in a Terminal. The golden audio is GuitarSet (CC BY 4.0,
https://guitarset.weebly.com/); the weights are for research and non-commercial use only (README, "Data
and licence").
