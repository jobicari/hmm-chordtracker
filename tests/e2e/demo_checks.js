// In-page checks of the demo's live mode (demo/app.js), run by tests/test_demo_e2e.py after a file has been
// imported. Each returns plain data for the test to assert on.
//
//   demoChecks.alignment(seekTo)   pause, seek to seekTo, play: where the player really is, found by matching
//                                  a snippet of its output against the browser's own decoding of the
//                                  imported file, against what the page believes (audio.currentTime) and
//                                  where the live session starts. {offset, sessionFrom, elementTime}
//   demoChecks.cancelledDrag()     a drag on the timeline that the browser cancels, then pointer moves with
//                                  no button: {sessionsOpened, moved} (no seek may follow the cancel)
//   demoChecks.scrub(from, to)     a drag across the timeline: {sessionsOpened, at} (one session, on release)
//   demoChecks.decidedAfter(wait)  after `wait` s of playback: {knownEnd, heard, segments}
//   demoChecks.staleLiveJs()       import a WAV with live.js's decodeForModel as it was before it returned the
//                                  playback WAV (an older live.js kept in the browser's cache): {status}
"use strict";

window.demoChecks = (() => {
  const sockets = [];
  const Native = window.WebSocket;
  window.WebSocket = function (...args) { const ws = new Native(...args); sockets.push(ws); return ws; };
  Object.assign(window.WebSocket, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  let ctx = null, source = null;

  async function decodedOriginal() {
    // The imported file as the browser decodes it, at the capture context's rate.
    return ctx.decodeAudioData(await upload.file.arrayBuffer());
  }

  async function alignment(seekTo) {
    if (!ctx) {
      ctx = new AudioContext();
      source = ctx.createMediaElementSource(audio);
      source.connect(ctx.destination);
    }
    await ctx.resume();
    const ref = (await decodedOriginal()).getChannelData(0), sr = ctx.sampleRate;
    if (!audio.paused || view.preroll) $("play").click();
    seek(seekTo);
    await sleep(300);
    const chunks = [];
    const tap = ctx.createScriptProcessor(4096, 1, 1);
    tap.onaudioprocess = (e) => {
      if (!audio.paused) chunks.push({ data: Float32Array.from(e.inputBuffer.getChannelData(0)), t: audio.currentTime });
    };
    source.connect(tap);
    tap.connect(ctx.destination);
    $("play").click();
    await sleep(playbackDelay() * 1000 + 1500);
    source.disconnect(tap);
    tap.disconnect();
    if (chunks.length < 4) return { error: `captured ${chunks.length} blocks` };
    const { data, t } = chunks[chunks.length - 2];
    let best = -1, bestScore = -Infinity;                // t: the element's time when the block was output
    const lo = Math.max(0, Math.floor((t - 8) * sr)), hi = Math.min(ref.length - data.length, Math.floor((t + 8) * sr));
    for (let i = lo; i < hi; i++) {
      let c = 0;
      for (let j = 0; j < 2048; j += 4) c += data[j] * ref[i + j];
      if (c > bestScore) { bestScore = c; best = i; }
    }
    // The block ends where the element was when it was handed over: it started one block earlier.
    return { offset: best / sr - (t - data.length / sr), sessionFrom: view.live && view.live.from, elementTime: t };
  }

  function pointer(type, t, buttons, pointerId = 7) {
    const rect = rows.decoded.track.getBoundingClientRect();
    $("timeline").dispatchEvent(new PointerEvent(type, {
      bubbles: true, pointerId, pointerType: "mouse", button: 0, buttons,
      clientX: rect.left + (t / view.sample.duration) * rect.width, clientY: rect.top + 5,
    }));
  }

  async function cancelledDrag() {
    pointer("pointerdown", 5, 1);
    pointer("pointercancel", 5, 0);
    await sleep(200);
    const opened = sockets.length, at = audio.currentTime;
    for (let i = 0; i < 10; i++) { pointer("pointermove", 6 + i, 0); await sleep(30); }
    await sleep(200);
    return { sessionsOpened: sockets.length - opened, moved: Math.abs(audio.currentTime - at) > 1e-3 };
  }

  async function scrub(from, to) {
    const opened = sockets.length;
    pointer("pointerdown", from, 1);
    for (let i = 1; i <= 20; i++) { pointer("pointermove", from + ((to - from) * i) / 20, 1); await sleep(20); }
    const during = sockets.length - opened;
    pointer("pointerup", to, 0);
    await sleep(300);
    return { sessionsDuring: during, sessionsOpened: sockets.length - opened, at: audio.currentTime };
  }

  async function decidedAfter(wait) {
    await sleep(wait * 1000);
    const d = view.sample.duration;
    return { knownEnd: rows.decoded.knownEnd, heard: Math.min(audio.currentTime, d), segments: rows.decoded.segs.length,
             playing: !audio.paused };
  }

  async function staleLiveJs() {
    // 2 s of noise as a WAV file, imported with live.js's decodeForModel as it was before it returned the
    // playback WAV; openFile's error goes to the status line, as the page's own import does.
    const buffer = new AudioBuffer({ length: 88200, sampleRate: 44100 });
    buffer.getChannelData(0).forEach((_, i, x) => { x[i] = 0.2 * (Math.random() * 2 - 1); });
    const current = decodeForModel;
    decodeForModel = async (blob, rate) => { const d = await current(blob, rate); return { pcm: d.pcm, duration: d.duration }; };
    try {
      await openFile(new File([wavBlob(buffer)], "stale.wav", { type: "audio/wav" })).catch((e) => setStatus(e.message, true));
    } finally {
      decodeForModel = current;
    }
    return { status: $("status").textContent };
  }

  return { alignment, cancelledDrag, scrub, decidedAfter, staleLiveJs };
})();
