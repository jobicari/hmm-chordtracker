// Chord Tracker demo, Phase 1: play a prepared sample and follow its decoded chords.
//
// Data: demo/samples/<id>.json, written by scripts/demo_export.py. The page draws its "decoded"
// segments (HMM / fixed-lag Viterbi) and decides nothing itself: it only looks up which segment holds
// the playback time.
//
// Clock: audio.currentTime is the only time source. The playhead, the seek bar, the time display and
// the decoded chord are all derived from it, on every animation frame while playing and at once on
// pause, seek and load.
//
// Active segment: the one with start <= t < end. At a boundary the new chord is active; at or past the
// end of the recording the last segment stays active.
//
// No flicker: the timeline is built once per sample. During playback only the playhead's transform,
// the seek bar's value and texts whose content changes are written; the chord texts and the active
// highlight are touched only when the active segment changes.
"use strict";

const PX_PER_SECOND = 40;           // minimum scale; the timeline scrolls when the recording is longer
const PITCH = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };

const $ = (id) => document.getElementById(id);
const audio = $("audio");
const view = { sample: null, segs: [], divs: [], active: undefined, timeText: "", width: 0, seeking: false,
               audioUrl: null };

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

function segmentAt(segs, t) {
  // Index of the last segment with start <= t (binary search); -1 if t is before the first.
  let lo = 0, hi = segs.length - 1, found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (segs[mid].start <= t) { found = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return found;
}

function nextDifferent(segs, i) {
  const now = i >= 0 ? segs[i].name : null;
  for (let j = i + 1; j < segs.length; j++) if (segs[j].name !== now) return segs[j].name;
  return "-";
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

function buildTimeline(sample) {
  // Once per sample: the segment blocks and the ruler, positioned in % of the duration.
  const d = sample.duration;
  const track = $("track");
  track.replaceChildren();
  view.divs = view.segs.map((seg) => {
    const div = document.createElement("div");
    div.className = "seg" + (rootPitch(seg.chord) === null ? " nc" : "");
    div.style.left = `${(100 * seg.start) / d}%`;
    div.style.width = `${(100 * (seg.end - seg.start)) / d}%`;
    const bg = colour(seg.chord);
    if (bg) div.style.backgroundColor = bg;
    div.textContent = seg.name;
    div.title = `${seg.name} (${seg.chord}), ${seg.start.toFixed(2)} - ${seg.end.toFixed(2)} s`;
    track.appendChild(div);
    return div;
  });
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
  $("timeline").style.width = `${Math.max(d * PX_PER_SECOND, 0)}px`;
  view.active = undefined;
  layout();
}

function layout() {
  // On load and resize: timeline width, and which blocks are too narrow to print their name.
  const scroller = $("scroller");
  const d = view.sample ? view.sample.duration : 0;
  $("timeline").style.width = `${Math.max(d * PX_PER_SECOND, scroller.clientWidth)}px`;
  view.width = $("track").getBoundingClientRect().width;
  view.segs.forEach((seg, i) =>
    view.divs[i].classList.toggle("narrow", ((seg.end - seg.start) / d) * view.width < 8 + 7 * seg.name.length));
  render();
}

function render() {
  // Everything that depends on the playback time, from audio.currentTime only.
  const sample = view.sample;
  if (!sample) return;
  const d = sample.duration;
  const t = Math.min(audio.currentTime, d);
  const x = (t / d) * view.width;
  $("playhead").style.transform = `translateX(${x}px)`;
  if (!view.seeking) $("seek").value = t;

  const text = `${formatTime(t)} / ${formatTime(d)}`;
  if (text !== view.timeText) { view.timeText = text; $("time").textContent = text; }

  const i = segmentAt(view.segs, t);
  if (i !== view.active) {
    if (view.active >= 0) view.divs[view.active].classList.remove("active");
    if (i >= 0) view.divs[i].classList.add("active");
    view.active = i;
    $("now").textContent = i >= 0 ? view.segs[i].name : "";
    $("now").title = i >= 0 ? view.segs[i].chord : "";
    $("next").textContent = nextDifferent(view.segs, i);
  }

  if (!audio.paused) {
    // Keep the playhead in view when the timeline scrolls: jump only when it leaves the visible part.
    const scroller = $("scroller");
    if (x < scroller.scrollLeft || x > scroller.scrollLeft + scroller.clientWidth - 20) {
      scroller.scrollLeft = Math.max(0, x - scroller.clientWidth * 0.1);
    }
  }
}

function loop() {
  render();
  if (!audio.paused) requestAnimationFrame(loop);
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
  view.audioUrl = null;
  view.sample = null;
  view.segs = [];
  view.divs = [];
  view.active = undefined;
  view.timeText = "";
  $("track").replaceChildren();
  $("ruler").replaceChildren();
  $("now").textContent = "";
  $("next").textContent = "-";
  $("time").textContent = "0:00.0 / 0:00.0";
  $("seek").value = 0;
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
  if (!Array.isArray(sample.decoded) || sample.decoded.length === 0) {
    throw new Error(`${entry.title}: the decoded output is missing.`);
  }
  const response = await fetch(`samples/${sample.audio}`);
  if (!response.ok) throw new Error(`samples/${sample.audio}: HTTP ${response.status}`);
  const blob = await response.blob();
  if (generation !== loadGeneration) return;
  view.sample = sample;
  view.segs = sample.decoded;
  view.audioUrl = URL.createObjectURL(blob);
  $("seek").max = sample.duration;
  buildTimeline(sample);
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

function togglePlay() {
  if (audio.paused) audio.play().catch((e) => setStatus(`Playback failed: ${e.message}`, true));
  else audio.pause();
}

function seek(t) {
  if (!view.sample) return;
  audio.currentTime = Math.max(0, Math.min(t, view.sample.duration));
  render();
}

audio.addEventListener("canplay", () => {
  if (!view.sample) return;
  setControls(true);
  setStatus(`Ready: ${view.sample.title}`);
});
audio.addEventListener("play", () => {
  $("play").textContent = "Pause";
  setStatus(`Playing: ${view.sample.title}`);
  requestAnimationFrame(loop);
});
audio.addEventListener("pause", () => {
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

const seekBar = $("seek");
seekBar.addEventListener("pointerdown", () => { view.seeking = true; });
seekBar.addEventListener("input", () => seek(Number(seekBar.value)));
seekBar.addEventListener("change", () => { view.seeking = false; seek(Number(seekBar.value)); });
seekBar.addEventListener("pointerup", () => { view.seeking = false; });

const timeline = $("timeline");
let dragging = false;
const seekFromPointer = (e) => {
  if (!view.sample || $("play").disabled) return;
  const rect = $("track").getBoundingClientRect();
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
