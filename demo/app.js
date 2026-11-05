// Chord Tracker demo: a prepared sample played back with its chords shown as the live system shows them.
//
// Data: demo/samples/<id>.json, written by scripts/demo_export.py. For each of the listening prototype's
// three operating points it holds the label segments and the delay after which a frame's label is known
// (decisionDelaySeconds: total look-ahead + one frame); the framewise row (network alone) has its own.
// A fixed-lag decision is final, so revealing each label when its delay has passed shows exactly what a
// live decoder would have shown. The page decides no chord itself.
//
// Two clocks, both from audio.currentTime (s, what you hear):
//   heard      = what the system has received. With "Delay playback" on, the audio really waits the
//                point's playback delay after Play, Restart or a seek while the input runs (pre-roll),
//                then heard = s + delay, also across pause / resume; with it off, heard = s
//   known(row) = up to the last frame f with f * h + the row's decision delay <= heard (h = one frame,
//                93 ms, so the mask moves one frame at a time, as the live output does); once the input
//                reaches the end of the recording the decoder flushes and everything is known
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

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const rows = {
  decoded: { track: $("track-decoded"), segs: [], divs: [], active: undefined, delay: 0, nowEl: $("now") },
  framewise: { track: $("track-framewise"), segs: [], divs: [], active: undefined, delay: 0, nowEl: $("now-framewise") },
};
const view = { sample: null, point: null, wantedPoint: null, timeText: "", width: 0, seeking: false, audioUrl: null,
               primed: false, preroll: null, internalPause: false };

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

function playbackDelay() {
  // With "Delay playback" on: the point's decision delay + half a frame, so that every label is
  // decided by the time its segment starts being heard (chordtracker.demo.playback_delay); else 0.
  return $("delay").checked && view.point ? view.point.playbackDelaySeconds : 0;
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
  row.divs = segs.map((seg) => {
    const div = document.createElement("div");
    div.className = "seg" + (rootPitch(seg.chord) === null ? " nc" : "");
    div.style.left = `${(100 * seg.start) / d}%`;
    div.style.width = `${(100 * (seg.end - seg.start)) / d}%`;
    const bg = colour(seg.chord);
    if (bg) div.style.backgroundColor = bg;
    div.textContent = seg.name;
    div.title = `${seg.name} (${seg.chord}), ${seg.start.toFixed(2)} - ${seg.end.toFixed(2)} s`;
    row.track.insertBefore(div, mask);
    return div;
  });
  markNarrow(row);
}

function markNarrow(row) {
  const d = view.sample ? view.sample.duration : 1;
  row.segs.forEach((seg, i) =>
    row.divs[i].classList.toggle("narrow", ((seg.end - seg.start) / d) * view.width < 8 + 7 * seg.name.length));
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
  buildRow(rows.decoded, point.segments, point.decisionDelaySeconds);
  const sub = document.createElement("span");
  sub.className = "sub";
  sub.textContent = `look-ahead ${point.lookaheadSeconds.toFixed(2)} s, ${point.segments.length} chord segments`;
  $("decoded-name").replaceChildren(
    point.pSelf === null ? "Decoded: no decoder, the network alone" : `Decoded: HMM / Viterbi, fixed lag of ${point.lagFrames} frames`,
    sub);
  updateDelayNote();
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
    title.textContent = `Applied delay: ${D} s. The audio is held back by ${D} s.`;
    text.textContent = `When you press Play, Restart or seek, the system starts listening at once and you hear `
      + `the music ${D} s later, as in a live set-up. By the time you hear a moment, the decoder has already `
      + `decided its chord, so every chord is shown when you hear it, exactly as a live decoder would show it.`;
    const items = [`${ms(networkFrames)} the network reads ahead (${networkFrames} frames)`];
    if (p.lagFrames > 0) items.push(`${ms(p.lagFrames)} the Viterbi decoder waits (lag of ${p.lagFrames} frames)`);
    items.push(`${ms(1)} for the last frame of audio to arrive`, `${ms(0.5)} so a chord is ready where its segment starts`);
    parts.textContent = `${D} s = ` + items.join(" + ") + ".";
  } else {
    title.textContent = "No delay: the audio plays at once.";
    text.textContent = `Each decoded chord is shown when a live decoder would output it, `
      + `${p.decisionDelaySeconds.toFixed(2)} s after you hear it (the network's own chord `
      + `${sample.framewiseDecisionDelaySeconds.toFixed(2)} s after). Turn the delay on to see them on time.`;
  }
  panel.replaceChildren(title, text, ...(parts.textContent ? [parts] : []));
  $("input-head").hidden = !$("delay").checked;
}

function layout() {
  // On load and resize: timeline width, and which blocks are too narrow to print their name.
  const d = view.sample ? view.sample.duration : 0;
  $("timeline").style.width = `${Math.max(d * PX_PER_SECOND, $("scroller").clientWidth)}px`;
  view.width = rows.decoded.track.getBoundingClientRect().width;
  if (view.sample) { markNarrow(rows.decoded); markNarrow(rows.framewise); }
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
  Object.assign(view, { sample: null, point: null, timeText: "", audioUrl: null, primed: false });
  for (const row of Object.values(rows)) {
    const mask = row.track.querySelector(".mask");
    mask.style.transform = "";
    row.track.replaceChildren(mask);
    Object.assign(row, { segs: [], divs: [], active: undefined });
    row.nowEl.textContent = "";
  }
  $("ruler").replaceChildren();
  $("decoded-name").textContent = "Decoded";
  $("delay-note").textContent = "";
  $("time").textContent = "0:00.0 / 0:00.0";
  $("seek").value = 0;
  $("points").disabled = true;
  setControls(false);
}

let loadGeneration = 0;

async function loadSample(entry) {
  // The audio is fetched whole and played from a blob URL: it is then seekable on any static server,
  // including python -m http.server, which does not answer byte-range requests. A newer load
  // (another sample picked meanwhile) makes this one drop its results.
  const generation = ++loadGeneration;
  clearSample();
  setStatus(`Loading ${entry.title}...`);
  const sample = await fetchJSON(`samples/${entry.file}`);
  if (!Array.isArray(sample.points) || sample.points.length === 0 || !Array.isArray(sample.framewise)) {
    throw new Error(`${entry.title}: the decoded or framewise output is missing.`);
  }
  for (const [name, segs] of [["framewise", sample.framewise], ...sample.points.map((p) => [p.id, p.segments])]) {
    if (!coversRecording(segs, sample.duration)) {
      throw new Error(`${entry.title}: the ${name} output does not cover this recording; it is not shown.`);
    }
  }
  const response = await fetch(`samples/${sample.audio}`);
  if (!response.ok) throw new Error(`samples/${sample.audio}: HTTP ${response.status}`);
  const blob = await response.blob();
  if (generation !== loadGeneration) return;
  view.sample = sample;
  view.audioUrl = URL.createObjectURL(blob);
  $("seek").max = sample.duration;
  $("timeline").style.width = `${Math.max(sample.duration * PX_PER_SECOND, $("scroller").clientWidth)}px`;
  view.width = rows.decoded.track.getBoundingClientRect().width;
  buildRuler(sample.duration);
  buildRow(rows.framewise, sample.framewise, sample.framewiseDecisionDelaySeconds);
  buildPoints(sample);
  $("points").disabled = false;
  selectPoint(view.wantedPoint || sample.defaultPoint);
  audio.src = view.audioUrl;
  audio.load();
  history.replaceState(null, "", `#${sample.id}`);
}

async function init() {
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
  const load = (entry) => {
    const generation = loadGeneration + 1;
    return loadSample(entry).catch((e) => {
      if (generation !== loadGeneration) return;          // a newer load replaced this one
      clearSample();
      setStatus(e.message, true);
    });
  };
  select.addEventListener("change", () => load(pick(select.value)));
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
  if (audio.ended) { audio.currentTime = 0; view.primed = false; }
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
  view.primed = false;
  if (running) { pauseQuietly(); play(); }
  render();
}

function togglePlay() {
  if (view.preroll) {                       // stopped during the pre-roll: the input stops too
    cancelPreroll();
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

const seekBar = $("seek");
seekBar.addEventListener("pointerdown", () => { view.seeking = true; });
seekBar.addEventListener("input", () => seek(Number(seekBar.value)));
seekBar.addEventListener("change", () => { view.seeking = false; seek(Number(seekBar.value)); });
seekBar.addEventListener("pointerup", () => { view.seeking = false; });

const timeline = $("timeline");
let dragging = false;
const seekFromPointer = (e) => {
  if (!view.sample || $("play").disabled) return;
  const rect = rows.decoded.track.getBoundingClientRect();
  seek(((e.clientX - rect.left) / rect.width) * view.sample.duration);
};
timeline.addEventListener("pointerdown", (e) => {
  dragging = true;
  seekFromPointer(e);
  try { timeline.setPointerCapture(e.pointerId); } catch { /* synthetic events have no capturable pointer */ }
});
timeline.addEventListener("pointermove", (e) => { if (dragging) seekFromPointer(e); });
timeline.addEventListener("pointerup", () => { dragging = false; });

document.addEventListener("keydown", (e) => {
  if (!view.sample || $("play").disabled || ["SELECT", "INPUT", "BUTTON"].includes(e.target.tagName)) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
});

init();
