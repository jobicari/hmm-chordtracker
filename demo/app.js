// Chord Tracker demo: a recording played back with its chords shown as the live system shows them.
//
// Two modes, both showing only what the live system would show:
//   live         scripts/demo_server.py is running: the page streams the audio to it at playback speed
//                (live.js) and draws each chord when the server sends it, decided by the live chain
//   precomputed  no server (a static host): demo/samples/<id>.json, written by scripts/demo_export.py,
//                holds every point's segments; each label is revealed once its decision delay has passed
//                (a fixed-lag decision is final, so this is what a live decoder would have shown)
// The page decides no chord itself.
//
// Two clocks, both from audio.currentTime (s, what you hear):
//   heard      = what the system has received. With "Delay playback" on, the audio really waits the
//                point's playback delay after Play, Restart or a seek while the input runs (pre-roll),
//                then heard = s + delay, also across pause / resume; with it off, heard = s
//   known(row) = up to the last frame f with f * h + the row's decision delay <= heard (h = one frame,
//                93 ms, so the mask moves one frame at a time, as the live output does); once the input
//                reaches the end of the recording the decoder flushes and everything is known
//                (precomputed); in live mode, up to the last frame the server has sent
// A row is revealed up to min(s, known): never past what you hear, never past what is decided. The
// chords shown "now" are the row's chord at that revealed time: on time with the delay, late without it.
//
// No flicker: tracks are built once per sample (and per look-ahead change, for the decoded row). During
// playback only transforms (playhead, input head, masks), the seek bar and changed texts are written; the
// chord texts and active highlights only when the displayed segment changes.
"use strict";

const PX_PER_SECOND = 40;           // minimum scale; the timeline scrolls when the recording is longer
const PITCH = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const EPS = 1e-4;
const LIVE_MARGIN = 0.15;           // s of playback delay for the network and the server, live mode only,
                                    // unless the server sets its own (liveMarginSeconds in api/health)
const MAX_UPLOAD_BYTES = 100 * 2 ** 20;

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const rows = {
  decoded: { track: $("track-decoded"), segs: [], divs: [], active: undefined, delay: 0, nowEl: $("now") },
  framewise: { track: $("track-framewise"), segs: [], divs: [], active: undefined, delay: 0, nowEl: $("now-framewise") },
};
// Reference annotations: not model output, so never masked; the active block follows what you hear.
const reference = { track: $("track-reference"), segs: [], divs: [], active: undefined, nowEl: $("now-reference") };
const MAJMIN = new Set(["maj", "min"]);
const view = { sample: null, point: null, wantedPoint: null, timeText: "", width: 0, seeking: false, audioUrl: null,
               primed: false, preroll: null, internalPause: false, mode: "precomputed", server: null, pcm: null,
               live: null };

// ---------------------------------------------------------------------------------------------------
// Chords

function rootPitch(chord) {
  // Harte label -> pitch class of its root; null for N / X. "F#:min7" -> 6.
  if (chord === "N" || chord === "X") return null;
  const root = chord.split(":")[0].split("/")[0];
  let pc = PITCH[root[0]];
  for (const ch of root.slice(1)) pc += ch === "#" ? 1 : ch === "b" ? -1 : 0;
  return ((pc % 12) + 12) % 12;
}

function colour(chord) {
  // Hue by root on the circle of fifths; minor-like qualities darker. Text labels carry the meaning.
  const pc = rootPitch(chord);
  if (pc === null) return null;
  const quality = (chord.split(":")[1] || "maj").split("/")[0];
  const minor = quality.startsWith("min") || quality.includes("dim");
  return `hsl(${((pc * 7) % 12) * 30}, ${minor ? 45 : 60}%, ${minor ? 33 : 45}%)`;
}

function chordName(label) {
  // "C:maj" -> "C", "A:min" -> "Am" (the model's 24 labels), "N" -> "N.C.", as chordtracker.play_along.chord_name.
  if (label === "N" || label === "X") return "N.C.";
  const [root, q = "maj"] = label.split(":");
  return root + (q === "min" ? "m" : "");
}

function coversRecording(segs, d) {
  // Same recording and run: every row starts at 0, has no gap or overlap, and ends at the audio's end
  // (the export cuts all of them from the same frame grid, chordtracker.demo.result).
  if (segs.length === 0 || segs[0].start !== 0 || Math.abs(segs.at(-1).end - d) > 1e-3) return false;
  return segs.every((g, i) => g.end > g.start && (i === 0 || Math.abs(g.start - segs[i - 1].end) < 1e-6));
}

function segmentAt(segs, t) {
  // Index of the last segment with start <= t (binary search); -1 if t is before the first.
  let lo = 0, hi = segs.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segs[mid].start <= t) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return found;
}

// ---------------------------------------------------------------------------------------------------
// Clocks

function liveMargin() {
  return view.server?.liveMarginSeconds ?? LIVE_MARGIN;
}

function playbackDelay() {
  // With "Delay playback" on: the point's decision delay + half a frame, so that every label is
  // decided by the time its segment starts being heard (chordtracker.demo.playback_delay); else 0.
  if (!$("delay").checked || !view.point) return 0;
  return view.point.playbackDelaySeconds + (view.mode === "live" ? liveMargin() : 0);
}

function heardTime(s, d) {
  // What the system has received. Delay on: after a start (Play from a fresh position, Restart, a seek)
  // the input runs alone for the delay while the audio waits (pre-roll), then stays that far ahead of
  // the audio, also across pause / resume. Delay off: the input is what you hear.
  const D = playbackDelay();
  if (view.preroll) return Math.min(view.preroll.from + Math.min((performance.now() - view.preroll.t0) / 1000, D), d);
  return Math.min(view.primed && D > 0 ? s + D : s, d);
}

function knownUntil(row, heard, d) {
  // End of the part of the row the live system has decided once it has received `heard` seconds.
  // Live mode: what the server has sent so far.
  if (view.mode === "live") return row.knownEnd ?? 0;
  // Frame f (centre f * h, region (f - 1/2) h .. (f + 1/2) h, as the segments are cut) is decided when
  // heard >= f * h + the row's decision delay; at the end of the input the decoder flushes the rest.
  const h = view.sample.metadata.frameSeconds;
  if (heard >= d) return d;
  if (heard < row.delay) return 0;
  return Math.min(d, (Math.floor((heard - row.delay) / h + 1e-6) + 0.5) * h);
}

function revealedUntil(row, s, heard, d) {
  // Shown: what is decided and what you have reached, min(s, known); the mask covers the rest.
  return Math.max(0, Math.min(s, knownUntil(row, heard, d)));
}

// ---------------------------------------------------------------------------------------------------
// Rendering

function formatTime(t) {
  const m = Math.floor(t / 60), s = t - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

function buildRow(row, segs, delay) {
  // Once per sample (and per look-ahead for the decoded row): the blocks, in % of the duration.
  const d = view.sample.duration;
  const mask = row.track.querySelector(".mask");
  row.track.replaceChildren(mask);
  row.segs = segs;
  row.delay = delay;
  row.active = undefined;
  row.knownEnd = 0;
  row.divs = segs.map((seg) => {
    const div = segmentDiv(seg, d);
    row.track.insertBefore(div, mask);
    return div;
  });
  markNarrow(row);
}

function segmentDiv(seg, d) {
  const div = document.createElement("div");
  div.className = "seg" + (rootPitch(seg.chord) === null ? " nc" : "");
  div.style.backgroundColor = colour(seg.chord) || "";
  div.textContent = seg.name;
  placeSegment(div, seg, d);
  return div;
}

function placeSegment(div, seg, d) {
  div.style.left = `${(100 * seg.start) / d}%`;
  div.style.width = `${(100 * (seg.end - seg.start)) / d}%`;
  div.title = `${seg.name} (${seg.chord}), ${seg.start.toFixed(2)} - ${seg.end.toFixed(2)} s`;
  div.classList.toggle("narrow", ((seg.end - seg.start) / d) * view.width < 8 + 7 * seg.name.length);
}

function appendLabels(row, from, labels) {
  // Live mode: frames from, from + 1, ... of the session arrived. Frame f stands for the time from
  // halfway after frame f - 1 to halfway before frame f + 1, as the segments of the precomputed mode;
  // a frame with the previous frame's chord extends its block, so only the last block is touched.
  const d = view.sample.duration, h = view.sample.metadata.frameSeconds, start = view.live.from;
  const mask = row.track.querySelector(".mask");
  labels.forEach((label, k) => {
    const f = from + k;
    const t0 = f === 0 ? start : start + (f - 0.5) * h;
    const t1 = Math.min(d, start + (f + 0.5) * h);
    if (t0 >= d) return;
    const chord = view.live.labelNames[label];
    const last = row.segs.at(-1);
    if (last && last.chord === chord && Math.abs(last.end - t0) < 1e-6) {
      last.end = t1;
      placeSegment(row.divs.at(-1), last, d);
    } else {
      const seg = { start: t0, end: t1, chord, name: chordName(chord) };
      row.segs.push(seg);
      const div = segmentDiv(seg, d);
      row.track.insertBefore(div, mask);
      row.divs.push(div);
    }
    row.knownEnd = t1;
  });
}

function markNarrow(row) {
  const d = view.sample ? view.sample.duration : 1;
  row.segs.forEach((seg, i) =>
    row.divs[i].classList.toggle("narrow", ((seg.end - seg.start) / d) * view.width < 8 + 7 * seg.name.length));
}

function quality(chord) {
  // "G#:min7/5" -> "min7"; "A" -> "maj"; "N" -> null.
  if (chord === "N" || chord === "X") return null;
  return (chord.split(":")[1] || "maj").split("/")[0].split("(")[0] || "maj";
}

function buildReference(sample) {
  // The reference row, or a note that there is none. Out-of-vocabulary chords are striped.
  const box = $("show-reference");
  reference.track.replaceChildren();
  reference.active = undefined;
  if (!Array.isArray(sample.reference) || sample.reference.length === 0) {
    reference.segs = [];
    reference.divs = [];
    box.checked = false;
    box.disabled = true;
    $("reference-note").textContent = "Reference annotations are not available for this recording.";
    showReference();
    return;
  }
  const d = sample.duration;
  reference.segs = sample.reference;
  reference.divs = sample.reference.map((seg) => {
    const div = document.createElement("div");
    const q = quality(seg.chord);
    div.className = "seg ref" + (q === null ? " nc" : "") + (q !== null && !MAJMIN.has(q) ? " out" : "");
    div.style.left = `${(100 * seg.start) / d}%`;
    div.style.width = `${(100 * (Math.min(seg.end, d) - seg.start)) / d}%`;
    const bg = colour(seg.chord);
    if (bg) div.style.setProperty("--root", bg);
    div.textContent = seg.name;
    div.title = `Reference: ${seg.name} (${seg.chord}), ${seg.start.toFixed(2)} - ${seg.end.toFixed(2)} s`;
    reference.track.appendChild(div);
    return div;
  });
  box.disabled = false;
  $("reference-sub").textContent = "reference labels, not model output";
  showReference();
}

function showReference() {
  const on = $("show-reference").checked && reference.segs.length > 0;
  document.querySelectorAll(".ref-only").forEach((el) => { el.hidden = !on; });
  if (reference.segs.length > 0) {
    $("reference-note").textContent = on
      ? `Reference: ${view.sample.referenceSource}. They show the chords written for the take, `
        + `aligned to this recording; the system was not given them.`
      : "";
  }
  if (on) markNarrow(reference);
  render();
}

function renderReference(s) {
  if (reference.segs.length === 0) return;
  const i = segmentAt(reference.segs, s);
  if (i === reference.active) return;
  if (reference.active >= 0) reference.divs[reference.active].classList.remove("active");
  if (i >= 0) reference.divs[i].classList.add("active");
  reference.active = i;
  reference.nowEl.textContent = i >= 0 ? reference.segs[i].name : "";
  reference.nowEl.title = i >= 0 ? reference.segs[i].chord : "";
}

const SVG = "http://www.w3.org/2000/svg";

function svg(tag, attrs = {}, text = null) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== null) el.textContent = text;
  return el;
}

function buildDiagram() {
  // The pipeline as a flow: four shared stages, then two branches (decoder, argmax) to the two rows.
  // Every value comes from the sample's metadata and the chosen look-ahead.
  const m = view.sample.metadata, P = m.pipeline, p = view.point;
  const ms = Math.round(m.frameSeconds * 1000);
  const W = 150, H = 96, MAIN_Y = 105, TOP_Y = 14, BOT_Y = 196;
  const root = svg("svg", { viewBox: "0 0 1120 306", xmlns: SVG });
  const defs = svg("defs");
  const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7,
                                 orient: "auto-start-reverse" });
  marker.appendChild(svg("path", { d: "M0 0 L10 5 L0 10 z", class: "arrow-head" }));
  defs.appendChild(marker);
  root.appendChild(defs);
  const box = (x, y, w, kind, num, title, lines) => {
    root.appendChild(svg("rect", { x, y, width: w, height: H, rx: 10, class: `box ${kind}` }));
    if (num) root.appendChild(svg("text", { x: x + 12, y: y + 22, class: "num" }, num));
    root.appendChild(svg("text", { x: x + (num ? 28 : 12), y: y + 22, class: "title" }, title));
    lines.forEach((line, i) => root.appendChild(svg("text", { x: x + 12, y: y + 46 + i * 17, class: "text" }, line)));
  };
  const arrow = (d) => root.appendChild(svg("path", { d, class: "arrow", "marker-end": "url(#arrow)" }));
  const xs = [10, 195, 380, 565];
  box(xs[0], MAIN_Y, W, "", "1", "Audio", [`${P.sampleRate.toLocaleString("en")} Hz, mono`,
    `${m.hopLength.toLocaleString("en")}-sample blocks`, `one per ${ms} ms frame`]);
  box(xs[1], MAIN_Y, W, "", "2", "Harmonic part", [`STFT ${P.harmonic.nFft}, hop ${P.harmonic.stftHop}`,
    `past ${P.harmonic.medianFrames}-frame median`, "keeps sustained notes"]);
  box(xs[2], MAIN_Y, W, "", "3", "Log-CQT", [`${P.cqt.binsPerOctave} bins per octave`,
    P.silence && P.silence.dbfs !== null ? `tuning: ${P.cqt.tuningSeconds} s from onset`
      : `tuning: first ${P.cqt.tuningSeconds} s`,
    `log(1 + ${P.compression.gamma} x)`]);
  box(xs[3], MAIN_Y, W, "model", "4", "CRNN", [`${P.network.windowFrames} frames, ${P.network.inputLookaheadFrames} ahead`,
    `${P.cqt.networkLowest} to ${P.cqt.networkHighest}`, `causal GRU, ${P.network.classes} chords`]);
  for (let i = 0; i < 3; i++) arrow(`M${xs[i] + W} ${MAIN_Y + H / 2} H${xs[i + 1] - 3}`);
  const decoderOn = p.pSelf !== null;
  box(775, TOP_Y, 170, decoderOn ? "decoder" : "decoder off", "5", "HMM + Viterbi", decoderOn
    ? ["Billboard chord changes", `p_self ${p.pSelf}`, `decides ${p.lagFrames} frames later`]
    : ["not used at this", "look-ahead: the row is", "the network alone"]);
  box(775, BOT_Y, 170, "", "5'", "Argmax", ["most probable chord", "of each frame", "no smoothing"]);
  box(990, TOP_Y, 120, "out", "", "Decoded", ["row, shown", `${p.decisionDelaySeconds.toFixed(2)} s later`]);
  box(990, BOT_Y, 120, "out", "", "Framewise", ["row, shown", `${view.sample.framewiseDecisionDelaySeconds.toFixed(2)} s later`]);
  const cx = xs[3] + W, cy = MAIN_Y + H / 2;
  arrow(`M${cx} ${cy} H${cx + 25} V${TOP_Y + H / 2} H${775 - 3}`);
  arrow(`M${cx} ${cy} H${cx + 25} V${BOT_Y + H / 2} H${775 - 3}`);
  arrow(`M945 ${TOP_Y + H / 2} H${990 - 3}`);
  arrow(`M945 ${BOT_Y + H / 2} H${990 - 3}`);
  $("pipeline-diagram").replaceChildren(root);
}

function buildPipeline() {
  // The stages in the order the code runs them, with the values of this run (metadata.pipeline).
  const m = view.sample.metadata, P = m.pipeline, p = view.point;
  const ms = Math.round(m.frameSeconds * 1000);
  const gated = Boolean(P.silence && P.silence.dbfs !== null);
  const stages = [
    ["Audio input", `The recording, mono, resampled to ${P.sampleRate.toLocaleString("en")} Hz. A live `
      + `system receives it in blocks of ${m.hopLength.toLocaleString("en")} samples (${ms} ms), one per frame.`],
    ["Harmonic part, from past audio only", `A short-time Fourier transform (${P.harmonic.nFft} samples, `
      + `hop ${P.harmonic.stftHop}) splits sustained notes from drum-like hits: a note counts as sustained `
      + `from the median of its last ${P.harmonic.medianFrames} STFT frames (margin ${P.harmonic.margin}). `
      + `This needs ${P.harmonic.lookaheadFrames} frame of future audio.`],
    ["Log constant-Q spectrum", `${P.cqt.binsPerOctave} bins per octave (${P.cqt.nBins} bins), one column `
      + `per ${ms} ms frame, each bin delayed so it reads only past audio. Tuning is estimated on the first `
      + `${P.cqt.tuningSeconds} s${gated ? ` of sound, from the first frame at or above ${P.silence.dbfs} dBFS `
        + "(a silent intro does not count)" : ""}; values are divided by the running ${P.compression.levelPercentile}th `
      + `percentile and compressed as log(1 + ${P.compression.gamma} x). Computed column by column, as live.`],
    ["Acoustic model (CRNN)", `For each frame, a window of ${P.network.windowFrames} columns `
      + `(${P.cqt.networkBins} bins, ${P.cqt.networkLowest} to ${P.cqt.networkHighest}) that ends `
      + `${P.network.inputLookaheadFrames} frames ahead goes through convolution layers; a GRU `
      + `(${P.network.gruHidden} units) then reads the frames in order, past only. Output: a probability for `
      + `each of ${P.network.classes} chords (12 major, 12 minor; there is no "no chord"). Model ${P.network.model}.`],
    ["Framewise output", "The most probable chord of each frame, alone. This is the Framewise row: it "
      + "changes whenever the network's top choice does, hence the flicker."],
    ["HMM", `${P.hmm.states} states, one per chord. Evidence: the network's probability of a chord divided `
      + `by how often it occurred in training. Transitions: stay on the chord with probability p_self, or `
      + `move as in chord-change counts from ${P.hmm.changeCountsSource} (alpha ${P.hmm.alpha}, tau `
      + `${P.hmm.tau}), the same for every key.`],
    ["Fixed-lag Viterbi", p.pSelf === null
      ? "Not used at the 0.28 s look-ahead: the decoded row is the network alone. Pick 0.46 s or 0.65 s to "
        + "add the decoder."
      : `The most likely chord path, decided with a lag of ${p.lagFrames} frames: frame t's chord is fixed `
        + `once frame t + ${p.lagFrames} has arrived and never changes afterwards. p_self ${p.pSelf}. `
        + `This is the Decoded row.`],
    ...(gated ? [["Silence gate", `A frame is shown as N.C. (no chord) on both rows when its own audio and `
      + `that of the ${P.silence.frames - 1} frames before it are below ${P.silence.dbfs} dBFS, so short rests keep `
      + `their chord. The network and the decoder keep running underneath: the chord comes back with the `
      + `first audible frame.`]] : []),
    ["Segments and display", `Runs of one chord become segments, cut halfway between frames. Each chord `
      + `appears once decided: ${p.decisionDelaySeconds.toFixed(2)} s after its frame for the decoded row, `
      + `${view.sample.framewiseDecisionDelaySeconds.toFixed(2)} s for the framewise row.`],
  ];
  buildDiagram();
  $("pipeline-stages").replaceChildren(...stages.map(([title, text]) => {
    const li = document.createElement("li");
    const b = document.createElement("strong");
    b.textContent = title + ". ";
    li.append(b, text);
    return li;
  }));
}

function buildRuler(d) {
  const ruler = $("ruler");
  ruler.replaceChildren();
  const step = d > 240 ? 30 : d > 60 ? 10 : 5;
  for (let t = 0; t <= d; t += step) {
    const tick = document.createElement("div");
    tick.className = "tick" + (t === 0 ? " first" : "") + (d - t < step / 2 ? " last" : "");
    tick.style.left = `${(100 * t) / d}%`;
    const label = document.createElement("span");
    label.textContent = `${t} s`;
    tick.appendChild(label);
    ruler.appendChild(tick);
  }
}

function buildPoints(sample) {
  // The look-ahead choice: one radio per operating point in the JSON.
  const box = $("points");
  box.querySelectorAll("label").forEach((l) => l.remove());
  for (const p of sample.points) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "point";
    input.value = p.id;
    input.addEventListener("change", () => selectPoint(p.id));
    const span = document.createElement("span");
    span.className = "sub";
    span.textContent = p.pSelf === null ? "network alone" : `Viterbi, lag ${p.lagFrames}`;
    label.append(input, `${p.lookaheadSeconds.toFixed(2)} s `, span);
    label.title = p.decoder;
    box.appendChild(label);
  }
}

function selectPoint(id) {
  const sample = view.sample;
  if (!sample) return;
  const point = sample.points.find((p) => p.id === id) || sample.points.find((p) => p.id === sample.defaultPoint);
  const changed = view.point !== null && view.point !== point;
  view.point = point;
  view.wantedPoint = point.id;
  $("points").querySelector(`input[value="${point.id}"]`).checked = true;
  buildRow(rows.decoded, view.mode === "live" ? [] : point.segments, point.decisionDelaySeconds);
  const sub = document.createElement("span");
  sub.className = "sub";
  sub.textContent = view.mode === "live" ? `look-ahead ${point.lookaheadSeconds.toFixed(2)} s`
    : `look-ahead ${point.lookaheadSeconds.toFixed(2)} s, ${point.segments.length} chord segments`;
  $("decoded-name").replaceChildren(
    point.pSelf === null ? "Decoded: no decoder, the network alone" : `Decoded: HMM / Viterbi, fixed lag of ${point.lagFrames} frames`,
    sub);
  updateDelayNote();
  buildPipeline();
  if (changed) restartInput(); else render();
}

function updateDelayNote() {
  // The delay panel: the delay applied, what it does, and what it is made of, all from the JSON.
  const panel = $("delay-note");
  if (!view.point) { panel.replaceChildren(); return; }
  const p = view.point, sample = view.sample;
  const h = sample.metadata.frameSeconds;
  const ms = (frames) => `${Math.round(frames * h * 1000)} ms`;
  const title = document.createElement("strong");
  const text = document.createElement("p");
  const parts = document.createElement("p");
  parts.className = "parts";
  if ($("delay").checked) {
    const D = p.playbackDelaySeconds.toFixed(2);
    const networkFrames = Math.round(p.lookaheadSeconds / h) - p.lagFrames;
    title.textContent = `Applied delay: the audio is held back by ${D} s.`;
    text.textContent = `After Play, Restart or a seek, the system listens ${D} s ahead of what you hear, `
      + `so each chord is shown as you hear it, exactly as a live decoder would show it.`;
    const items = [`network look-ahead ${ms(networkFrames)}`];
    if (p.lagFrames > 0) items.push(`decoder lag ${ms(p.lagFrames)}`);
    items.push(`last frame of audio ${ms(1)}`, `half a frame ${ms(0.5)}`);
    if (view.mode === "live") items.push(`network and server margin ${Math.round(liveMargin() * 1000)} ms`);
    parts.textContent = `${(playbackDelay()).toFixed(2)} s = ` + items.join(" + ");
    if (view.mode === "live") {
      title.textContent = `Applied delay: the audio is held back by ${playbackDelay().toFixed(2)} s.`;
      text.textContent = `After Play, Restart or a seek, your browser streams the recording to the demo `
        + `server, which decodes it live; you hear it ${playbackDelay().toFixed(2)} s later, so each chord `
        + `is on screen as you hear it.`;
    }
  } else {
    title.textContent = "No delay: the audio plays at once.";
    text.textContent = `Chords now appear when a live decoder would output them: the decoded chord `
      + `${p.decisionDelaySeconds.toFixed(2)} s after you hear it, the network's own `
      + `${sample.framewiseDecisionDelaySeconds.toFixed(2)} s after.`;
  }
  const measured = document.createElement("p");
  measured.className = "parts";
  measured.id = "measured";
  panel.replaceChildren(title, text, ...(parts.textContent ? [parts] : []), ...(view.mode === "live" ? [measured] : []));
  showMeasured();
  $("input-head").hidden = !$("delay").checked;
}

function showMeasured() {
  // Live mode: when this session's chords reached the page, after their moment of audio.
  const el = $("measured");
  if (!el || !view.live) return;
  const a = view.live.arrival;
  if (!a) { el.textContent = "This session: waiting for the first chords."; return; }
  const range = `${a.min.toFixed(2)} to ${a.max.toFixed(2)} s`;
  const budget = playbackDelay() - view.sample.metadata.frameSeconds / 2;
  el.textContent = $("delay").checked
    ? `This session: each chord reached the page ${range} after its moment of audio; the delay leaves `
      + `${budget.toFixed(2)} s, so every chord is on screen in time.`
    : `This session: each chord reached the page ${range} after its moment of audio, so you see it that late.`;
  if ($("delay").checked && a.max > budget) {
    el.textContent = `This session: some chords reached the page ${a.max.toFixed(2)} s after their moment of `
      + `audio, more than the ${budget.toFixed(2)} s the delay leaves: those were shown late.`;
  }
}

function layout() {
  // On load and resize: timeline width, and which blocks are too narrow to print their name.
  const d = view.sample ? view.sample.duration : 0;
  $("timeline").style.width = `${Math.max(d * PX_PER_SECOND, $("scroller").clientWidth)}px`;
  view.width = rows.decoded.track.getBoundingClientRect().width;
  if (view.sample) {
    for (const row of [rows.decoded, rows.framewise]) row.segs.forEach((seg, i) => placeSegment(row.divs[i], seg, d));
    markNarrow(reference);
  }
  render();
}

function renderRow(row, revealed) {
  row.track.querySelector(".mask").style.transform = `translateX(${(revealed / view.sample.duration) * view.width}px)`;
  const i = revealed > 0 ? segmentAt(row.segs, revealed - EPS) : -1;
  if (i === row.active) return;                                   // same chord: touch nothing
  if (row.active >= 0) row.divs[row.active].classList.remove("active");
  if (i >= 0) row.divs[i].classList.add("active");
  row.active = i;
  row.nowEl.textContent = i >= 0 ? row.segs[i].name : "";
  row.nowEl.title = i >= 0 ? row.segs[i].chord : "";
}

function render() {
  // Everything that depends on the playback time, from audio.currentTime only.
  const sample = view.sample;
  if (!sample || !view.point) return;
  const d = sample.duration;
  const s = Math.min(audio.currentTime, d);
  const heard = heardTime(s, d);
  const x = (s / d) * view.width;
  $("playhead").style.transform = `translateX(${x}px)`;
  $("input-head").style.transform = `translateX(${(heard / d) * view.width}px)`;
  if (!view.seeking) $("seek").value = s;
  const text = `${formatTime(s)} / ${formatTime(d)}`;
  if (text !== view.timeText) { view.timeText = text; $("time").textContent = text; }
  for (const row of Object.values(rows)) renderRow(row, revealedUntil(row, s, heard, d));
  renderReference(s);

  if (!audio.paused || view.preroll) {
    // Keep the playhead in view when the timeline scrolls: jump only when it leaves the visible part.
    const scroller = $("scroller");
    if (x < scroller.scrollLeft || x > scroller.scrollLeft + scroller.clientWidth - 20) {
      scroller.scrollLeft = Math.max(0, x - scroller.clientWidth * 0.1);
    }
  }
}

function loop() {
  render();
  if (!audio.paused || view.preroll) requestAnimationFrame(loop);
}

// ---------------------------------------------------------------------------------------------------
// Loading

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

function setControls(enabled) {
  for (const id of ["play", "restart", "seek"]) $(id).disabled = !enabled;
}

function clearSample() {
  // Nothing of the previous sample may stay on screen while the next one loads.
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
  if (view.audioUrl) URL.revokeObjectURL(view.audioUrl);
  cancelPreroll();
  stopSession();
  document.body.classList.remove("analyzing");
  Object.assign(view, { sample: null, point: null, timeText: "", audioUrl: null, primed: false, pcm: null });
  for (const row of Object.values(rows)) {
    const mask = row.track.querySelector(".mask");
    mask.style.transform = "";
    row.track.replaceChildren(mask);
    Object.assign(row, { segs: [], divs: [], active: undefined });
    row.nowEl.textContent = "";
  }
  $("ruler").replaceChildren();
  reference.track.replaceChildren();
  Object.assign(reference, { segs: [], divs: [], active: undefined });
  reference.nowEl.textContent = "";
  $("reference-note").textContent = "";
  $("pipeline-stages").replaceChildren();
  $("pipeline-diagram").replaceChildren();
  $("decoded-name").textContent = "Decoded";
  $("delay-note").textContent = "";
  $("time").textContent = "0:00.0 / 0:00.0";
  $("seek").value = 0;
  $("points").disabled = true;
  setControls(false);
}

let loadGeneration = 0;
const upload = { file: null, sample: null, pcm: null, playback: null };   // the listener's own file, decoded once

function checkSample(sample, title) {
  // Refuse outputs that are missing or do not cover the recording: nothing partial is shown.
  if (!Array.isArray(sample.points) || sample.points.length === 0 || !Array.isArray(sample.framewise)) {
    throw new Error(`${title}: the decoded or framewise output is missing.`);
  }
  for (const [name, segs] of [["framewise", sample.framewise], ...sample.points.map((p) => [p.id, p.segments])]) {
    if (!coversRecording(segs, sample.duration)) {
      throw new Error(`${title}: the ${name} output does not cover this recording; it is not shown.`);
    }
  }
}

function showSample(sample, audioBlob, pcm) {
  // Build everything for one recording and its audio (a Blob: the sample's WAV, or an upload's decoded
  // audio as WAV); pcm for live mode.
  view.sample = sample;
  view.pcm = pcm;
  view.audioUrl = URL.createObjectURL(audioBlob);
  $("seek").max = sample.duration;
  $("timeline").style.width = `${Math.max(sample.duration * PX_PER_SECOND, $("scroller").clientWidth)}px`;
  view.width = rows.decoded.track.getBoundingClientRect().width;
  buildRuler(sample.duration);
  buildRow(rows.framewise, view.mode === "live" ? [] : sample.framewise, sample.framewiseDecisionDelaySeconds);
  buildReference(sample);
  buildPoints(sample);
  $("points").disabled = false;
  selectPoint(view.wantedPoint || sample.defaultPoint);
  audio.src = view.audioUrl;
  audio.load();
}

async function loadSample(entry) {
  // A prepared sample. The audio is fetched whole and played from a blob URL: it is then seekable on
  // any static server, including python -m http.server, which does not answer byte-range requests.
  // Live mode also decodes it for streaming. A newer load (another sample or a file picked meanwhile)
  // makes this one drop its results.
  const generation = ++loadGeneration;
  clearSample();
  $("file-name").textContent = "";
  setStatus(`Loading ${entry.title}...`);
  const sample = await fetchJSON(`samples/${entry.file}`);
  checkSample(sample, entry.title);
  const response = await fetch(`samples/${sample.audio}`);
  if (!response.ok) throw new Error(`samples/${sample.audio}: HTTP ${response.status}`);
  const blob = await response.blob();
  const pcm = view.mode === "live" ? (await decodeForModel(blob, view.server.metadata.sampleRate)).pcm : null;
  if (generation !== loadGeneration) return;
  showSample(sample, blob, pcm);
  history.replaceState(null, "", `#${sample.id}`);
}

async function openFile(file) {
  // The listener's own file: decoded by the browser, streamed live to the demo server as it plays.
  const generation = ++loadGeneration;
  clearSample();
  upload.file = file;
  upload.sample = null;
  upload.pcm = null;
  upload.playback = null;
  setUploadOption(file.name);
  $("file-name").textContent = file.name;
  if (file.size === 0) throw new Error(`${file.name}: the file is empty.`);
  if (file.size > MAX_UPLOAD_BYTES) throw new Error(`${file.name}: larger than ${MAX_UPLOAD_BYTES / 2 ** 20} MB.`);
  setStatus(`Decoding ${file.name}...`);
  document.body.classList.add("analyzing");
  let decoded;
  try {
    decoded = await decodeForModel(file, view.server.metadata.sampleRate);
  } catch {
    throw new Error(`${file.name}: this browser cannot decode it as audio.`);
  }
  if (generation !== loadGeneration) return;
  document.body.classList.remove("analyzing");
  if (decoded.duration < 1 || decoded.duration > view.server.maxSeconds) {
    throw new Error(`${file.name}: ${decoded.duration.toFixed(1)} s long; the demo takes 1 s to `
      + `${view.server.maxSeconds / 60} min.`);
  }
  const s = view.server;
  upload.sample = {
    id: "upload", title: file.name, audio: null, duration: decoded.duration,
    points: s.points.map((p) => ({ ...p, segments: [] })), defaultPoint: s.defaultPoint,
    framewise: [], framewiseDecisionDelaySeconds: s.framewiseDecisionDelaySeconds,
    reference: null, referenceSource: null, metadata: s.metadata, source: { dataset: "your file" },
  };
  upload.pcm = decoded.pcm;
  upload.playback = decoded.playback;          // played instead of the file: exact seeks (live.js)
  showSample(upload.sample, upload.playback, decoded.pcm);
  history.replaceState(null, "", "#upload");
}

function setUploadOption(name) {
  // The uploaded file appears in the sample list, selected, so the list never names another recording.
  const select = $("sample");
  let option = select.querySelector('option[value="upload"]');
  if (!option) { option = new Option("", "upload"); select.add(option); }
  option.textContent = `Your file: ${name}`;
  select.value = "upload";
}

async function initServer() {
  // Live mode when scripts/demo_server.py answers; otherwise the precomputed samples only.
  try {
    view.server = await fetchJSON("api/health");
    if (!view.server.live) throw new Error("no live decoding");
    view.mode = "live";
    $("file").disabled = false;
    $("upload").classList.remove("unavailable");
    $("upload-help").textContent = `Any audio file your browser can play (WAV, MP3, FLAC, Ogg, M4A...), up to `
      + `${MAX_UPLOAD_BYTES / 2 ** 20} MB and ${view.server.maxSeconds / 60} min. Streamed to the demo server `
      + `and decoded live as it plays.`;
    $("notice").textContent = "Decoded live: your browser streams the recording to the demo server, which runs "
      + "the model as you listen and sends back each chord as soon as it is decided.";
  } catch {
    view.server = null;
    view.mode = "precomputed";
    $("upload-help").textContent = "Your own files need the demo server: python scripts/demo_server.py "
      + "(see demo/README.md).";
    $("notice").textContent = "The demo server is not running, so these chords were computed ahead of time. "
      + "Each one appears when the live system would have shown it, never earlier.";
  }
}

// ---------------------------------------------------------------------------------------------------
// Live sessions

function startSession() {
  // A new live session from what you hear now: the rows start empty and fill as the server decides.
  stopSession();
  const from = audio.currentTime;
  const session = new LiveSession({
    pcm: view.pcm, sampleRate: view.server.metadata.sampleRate, from,
    onReady: (names) => { if (view.live && view.live.session === session) view.live.labelNames = names; },
    onLabels: (point, first, labels) => {
      if (!view.live || view.live.session !== session) return;
      receiveLabels(point, first, labels);
    },
    onDone: () => {
      if (!view.live || view.live.session !== session) return;
      for (const row of Object.values(rows)) row.knownEnd = view.sample.duration;
      render();
    },
    onError: (message) => {
      if (!view.live || view.live.session !== session) return;
      stopSession();
      cancelPreroll();
      view.primed = false;
      pauseQuietly();
      $("play").textContent = "Play";
      setStatus(message, true);
    },
  });
  view.live = { session, from, labelNames: [], arrival: null, sender: null };
  for (const row of Object.values(rows)) buildRow(row, [], row.delay);
  rows.decoded.knownEnd = rows.framewise.knownEnd = from;
  // Stream on a timer, not on animation frames, so the audio keeps flowing when the tab is hidden.
  view.live.sender = setInterval(() => {
    if (!view.sample) return;
    const d = view.sample.duration;
    const heard = heardTime(Math.min(audio.currentTime, d), d);
    session.sendUntil(heard >= d - 1e-6 ? Infinity : heard);   // at the end, every sample (the decoded
  }, 20);                                                       // length can differ from the player's by a few)
  showMeasured();
}

function stopSession() {
  // End the live session; what it decided goes with it (a new session starts from scratch).
  if (!view.live) return;
  clearInterval(view.live.sender);
  view.live.session.close();
  view.live = null;
  if (view.sample) for (const row of Object.values(rows)) buildRow(row, [], row.delay);
}

function receiveLabels(point, first, labels) {
  const framewise = view.sample.points.find((p) => p.pSelf === null);
  if (point === framewise.id) appendLabels(rows.framewise, first, labels);
  if (point === view.point.id) {
    appendLabels(rows.decoded, first, labels);
    // When each chord reached the page, counted from its frame's moment of audio (what the delay must cover).
    const h = view.sample.metadata.frameSeconds;
    const d = view.sample.duration;
    const heard = heardTime(Math.min(audio.currentTime, d), d);
    if (heard < d) {
      const after = heard - (view.live.from + (first + labels.length - 1) * h);
      const a = view.live.arrival;
      const before = a ? `${Math.round(a.min * 100)} ${Math.round(a.max * 100)}` : "";
      view.live.arrival = a ? { min: Math.min(a.min, after), max: Math.max(a.max, after) } : { min: after, max: after };
      const now = view.live.arrival;
      if (`${Math.round(now.min * 100)} ${Math.round(now.max * 100)}` !== before) showMeasured();   // 10 ms steps
    }
  }
}

async function init() {
  await initServer();
  let index;
  try {
    index = await fetchJSON("samples/index.json");
  } catch (e) {
    setStatus("No prepared samples found. Run scripts/demo_export.py first (see demo/README.md).", true);
    return;
  }
  const select = $("sample");
  for (const s of index.samples) select.add(new Option(s.title, s.id));
  select.disabled = false;
  const pick = (id) => index.samples.find((s) => s.id === id);
  const guard = (promise) => {
    const generation = loadGeneration;                    // the load just started
    return promise.catch((e) => {
      if (generation !== loadGeneration) return;          // a newer load replaced this one
      clearSample();
      setStatus(e.message, true);
    });
  };
  const load = (entry) => { const p = loadSample(entry); return guard(p); };
  const analyze = (file) => { const p = openFile(file); return guard(p); };
  select.addEventListener("change", () => {
    if (select.value === "upload") {
      if (upload.sample) {                                // back to the file decoded before
        ++loadGeneration;
        clearSample();
        showSample(upload.sample, upload.playback, upload.pcm);
      } else if (upload.file) {
        analyze(upload.file);
      }
    } else {
      load(pick(select.value));
    }
  });
  $("file").addEventListener("change", () => { if ($("file").files[0]) analyze($("file").files[0]); $("file").value = ""; });
  const zone = $("upload");
  zone.addEventListener("dragover", (e) => { if (!$("file").disabled) { e.preventDefault(); zone.classList.add("over"); } });
  zone.addEventListener("dragleave", () => zone.classList.remove("over"));
  zone.addEventListener("drop", (e) => {
    zone.classList.remove("over");
    if ($("file").disabled) return;
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length !== 1) { setStatus("Drop one audio file at a time.", true); return; }
    analyze(files[0]);
  });
  // A file dropped outside the zone must not make the browser leave the page.
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => e.preventDefault());
  const wanted = pick(location.hash.slice(1)) || index.samples[0];
  select.value = wanted.id;
  new ResizeObserver(layout).observe($("scroller"));
  await load(wanted);
}

// ---------------------------------------------------------------------------------------------------
// Controls

function startAudio() {
  audio.play().catch((e) => setStatus(`Playback failed: ${e.message}`, true));
}

function cancelPreroll() {
  if (!view.preroll) return;
  clearTimeout(view.preroll.timer);
  view.preroll = null;
}

function play() {
  // Start playback; with the delay on and the input not yet ahead, hold the audio back first.
  // Live mode: the input is a session on the demo server, started here if there is none.
  if (audio.ended) { audio.currentTime = 0; view.primed = false; stopSession(); }
  if (view.mode === "live" && !view.live) startSession();
  const D = playbackDelay();
  if (D === 0 || view.primed) { startAudio(); return; }
  cancelPreroll();
  view.preroll = { from: audio.currentTime, t0: performance.now(), timer: null };
  view.preroll.timer = setTimeout(() => {
    view.preroll = null;
    view.primed = true;
    startAudio();
  }, D * 1000);
  $("play").textContent = "Pause";
  setStatus(`Playback delayed ${D.toFixed(2)} s: the system is listening ahead...`);
  requestAnimationFrame(loop);
}

function pauseQuietly() {
  // Stop the audio without reporting "Paused" (the playback restarts at once).
  if (audio.paused) return;
  view.internalPause = true;
  audio.pause();
}

function restartInput() {
  // The input starts again from what you hear (a seek, a new delay or look-ahead): while playing,
  // pre-roll again; while paused, the next Play does.
  const running = !audio.paused || view.preroll;
  cancelPreroll();
  stopSession();
  view.primed = false;
  if (running) { pauseQuietly(); play(); }
  render();
}

function togglePlay() {
  if (view.preroll) {                       // stopped during the pre-roll: the input stops too
    cancelPreroll();
    stopSession();
    view.primed = false;
    $("play").textContent = "Play";
    setStatus(`Paused: ${view.sample.title}`);
    render();
  } else if (audio.paused) {
    play();
  } else {
    audio.pause();
  }
}

function seek(t) {
  if (!view.sample) return;
  audio.currentTime = Math.max(0, Math.min(t, view.sample.duration));
  restartInput();
}

// Scrubbing (dragging the seek bar or the timeline): the playhead and the audio follow the pointer, the
// input stops; one new input starts from where it is let go, so a drag starts one live session, not one
// per pointer move. A drag ends on release, on cancel (the browser took the gesture, e.g. to scroll) and
// on lost capture, so a gesture that never releases cannot leave the page seeking on every move.
const scrub = { active: false, running: false };

function scrubStart() {
  if (scrub.active || !view.sample) return;
  scrub.active = true;
  scrub.running = !audio.paused || Boolean(view.preroll);
  cancelPreroll();
  stopSession();
  view.primed = false;
  pauseQuietly();
}

function scrubTo(t) {
  if (!view.sample) return;
  audio.currentTime = Math.max(0, Math.min(t, view.sample.duration));
  render();
}

function scrubEnd() {
  if (!scrub.active) return;
  scrub.active = false;
  if (scrub.running) play();
  render();
}

audio.addEventListener("canplay", () => {
  if (!view.sample) return;
  setControls(true);
  if (audio.paused) setStatus(`Ready: ${view.sample.title}`);
});
audio.addEventListener("play", () => {
  $("play").textContent = "Pause";
  setStatus(`Playing: ${view.sample.title}`);
  requestAnimationFrame(loop);
});
audio.addEventListener("pause", () => {
  if (view.internalPause) { view.internalPause = false; return; }
  $("play").textContent = "Play";
  if (view.sample) setStatus(`${audio.ended ? "Ended" : "Paused"}: ${view.sample.title}`);
  render();
});
audio.addEventListener("seeked", render);
audio.addEventListener("ended", render);
audio.addEventListener("loadedmetadata", render);
audio.addEventListener("error", () => {
  if (view.sample) { setStatus(`Could not load the audio of ${view.sample.title}.`, true); setControls(false); }
});

$("play").addEventListener("click", togglePlay);
$("restart").addEventListener("click", () => seek(0));
$("delay").addEventListener("change", () => { updateDelayNote(); restartInput(); });
$("show-reference").addEventListener("change", showReference);

const seekBar = $("seek");
const seekBarDone = () => { view.seeking = false; scrubEnd(); };
seekBar.addEventListener("pointerdown", () => { view.seeking = true; scrubStart(); });
seekBar.addEventListener("input", () => {                      // a drag scrubs; a key press seeks at once
  if (scrub.active) scrubTo(Number(seekBar.value)); else seek(Number(seekBar.value));
});
seekBar.addEventListener("change", () => { if (scrub.active) { scrubTo(Number(seekBar.value)); seekBarDone(); } });
for (const type of ["pointerup", "pointercancel"]) seekBar.addEventListener(type, seekBarDone);

const timeline = $("timeline");
let dragging = null;                                           // the pointer id of the drag on the timeline
const timeAt = (e) => {
  const rect = rows.decoded.track.getBoundingClientRect();
  return ((e.clientX - rect.left) / rect.width) * view.sample.duration;
};
const usable = () => view.sample && !$("play").disabled;
timeline.addEventListener("pointerdown", (e) => {
  if (!usable() || e.button !== 0) return;
  if (e.pointerType === "touch") return;                       // touch: a swipe scrolls, a tap seeks (pointerup)
  dragging = e.pointerId;
  scrubStart();
  scrubTo(timeAt(e));
  try { timeline.setPointerCapture(e.pointerId); } catch { /* synthetic events have no capturable pointer */ }
});
timeline.addEventListener("pointermove", (e) => {
  if (dragging === e.pointerId && (e.buttons & 1)) scrubTo(timeAt(e));
});
const dragDone = (e) => {
  if (dragging === null || e.pointerId !== dragging) return;
  dragging = null;
  scrubEnd();
};
for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) timeline.addEventListener(type, dragDone);
let tap = null;                                                // touch: where a touch started, to tell a tap
timeline.addEventListener("pointerdown", (e) => { if (e.pointerType === "touch") tap = { id: e.pointerId, x: e.clientX }; });
timeline.addEventListener("pointercancel", (e) => { if (tap && tap.id === e.pointerId) tap = null; });
timeline.addEventListener("pointerup", (e) => {
  if (!tap || tap.id !== e.pointerId) return;
  const moved = Math.abs(e.clientX - tap.x);
  tap = null;
  if (moved < 10 && usable()) seek(timeAt(e));
});

document.addEventListener("keydown", (e) => {
  if (!view.sample || $("play").disabled || ["SELECT", "INPUT", "BUTTON"].includes(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
});

init();
