// Live decoding: decode a recording in the browser, stream it to scripts/demo_server.py at playback
// speed, and receive each chord as soon as the server's decoders have decided it.
//
//   decodeForModel(blob, sampleRate) -> {pcm, duration}: the browser's own decoder, mixed down to mono
//                                       and resampled to the model's rate (22,050 Hz)
//   new LiveSession({...})           -> one WebSocket session; sendUntil(t) streams the audio up to t
//                                       seconds of the recording; close() ends it
//
// The protocol is described in scripts/demo_server.py.
"use strict";

async function decodeForModel(blob, sampleRate) {
  const context = new AudioContext();
  let decoded;
  try {
    decoded = await context.decodeAudioData(await blob.arrayBuffer());
  } finally {
    context.close();
  }
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * sampleRate)), sampleRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);            // one output channel: the channels are averaged
  source.start();
  const rendered = await offline.startRendering();
  return { pcm: rendered.getChannelData(0), duration: decoded.duration };
}

class LiveSession {
  // pcm: Float32Array at sampleRate; from: the second of the recording the session starts at.
  // Callbacks: onReady(labelNames), onLabels(pointId, firstFrame, labels), onDone(), onError(message).
  constructor({ pcm, sampleRate, from, onReady, onLabels, onDone, onError }) {
    Object.assign(this, { pcm, sampleRate, from, onReady, onLabels, onDone, onError });
    this.next = Math.round(from * sampleRate);    // next sample to send
    this.ready = false;
    this.ended = false;
    this.finished = false;
    this.closed = false;
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${scheme}://${location.host}/api/live`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onopen = () => this.ws.send(JSON.stringify({ type: "start" }));
    this.ws.onmessage = (event) => this.receive(JSON.parse(event.data));
    this.ws.onclose = () => {
      if (!this.closed && !this.finished) this.onError("The connection to the demo server was lost.");
    };
  }

  receive(message) {
    if (this.closed) return;
    if (message.type === "ready") {
      this.ready = true;
      this.onReady(message.labels);
    } else if (message.type === "labels") {
      for (const [point, value] of Object.entries(message.points)) this.onLabels(point, value.from, value.labels);
    } else if (message.type === "done") {
      this.finished = true;
      this.onDone();
    } else if (message.type === "error") {
      this.finished = true;
      this.onError(message.message);
    }
  }

  sendUntil(seconds) {
    // Stream the samples up to `seconds` of the recording (never ahead of what the system has heard).
    if (!this.ready || this.ended || this.closed || this.ws.readyState !== WebSocket.OPEN) return;
    const target = Math.min(this.pcm.length, Math.floor(seconds * this.sampleRate));
    while (this.next < target) {
      const end = Math.min(target, this.next + 8192);
      const block = new Int16Array(end - this.next);
      for (let i = 0; i < block.length; i++) {
        const x = this.pcm[this.next + i];
        block[i] = Math.max(-32768, Math.min(32767, Math.round(x * 32767)));
      }
      this.ws.send(block.buffer);
      this.next = end;
    }
    if (this.next >= this.pcm.length) {
      this.ws.send(JSON.stringify({ type: "end" }));
      this.ended = true;
    }
  }

  close() {
    this.closed = true;
    try { this.ws.close(); } catch { /* already closed */ }
  }
}
