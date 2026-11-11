// Live decoding: decode a recording in the browser, stream it to scripts/demo_server.py at playback
// speed, and receive each chord as soon as the server's decoders have decided it.
//
//   decodeForModel(blob, sampleRate) -> {pcm, duration, playback}: the browser's own decoder, mixed down
//                                       to mono and resampled to the model's rate (22,050 Hz); playback:
//                                       the same decoded audio as a WAV Blob, for the player (wavBlob)
//   wavBlob(audioBuffer)             -> 16-bit PCM WAV of every channel, at the buffer's rate
//   new LiveSession({...})           -> one WebSocket connection; sendUntil(t) streams the audio up to t
//                                       seconds of the recording; restart({...}) starts a fresh session
//                                       on the same connection (a seek); close() ends it. The server
//                                       closes it after "done" and when no audio came for a while ("idle")
//
// The protocol is described in scripts/demo_server.py.
"use strict";

async function decodeForModel(blob, sampleRate) {
  // The player plays `playback`, not the original file: a compressed file (MP3, FLAC, AAC...) seeks only
  // approximately in some browsers (Safari's FLAC and MP3), so after a seek the player would play other
  // samples than the ones streamed from the same time. An uncompressed WAV seeks exactly everywhere, and
  // being the same decoded audio, it has the same samples and the same length as what the server gets.
  const context = new AudioContext();
  let decoded;
  try {
    decoded = await context.decodeAudioData(await blob.arrayBuffer());
  } finally {
    context.close();
  }
  const playback = wavBlob(decoded);
  const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * sampleRate)), sampleRate);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);            // one output channel: the channels are averaged
  source.start();
  const rendered = await offline.startRendering();
  return { pcm: rendered.getChannelData(0), duration: decoded.duration, playback };
}

function wavBlob(buffer) {
  // A 16-bit PCM WAV (RIFF, little-endian) of an AudioBuffer: channels interleaved, at buffer.sampleRate.
  const channels = buffer.numberOfChannels, frames = buffer.length, rate = buffer.sampleRate;
  const header = new DataView(new ArrayBuffer(44));
  const text = (offset, s) => { for (let i = 0; i < s.length; i++) header.setUint8(offset + i, s.charCodeAt(i)); };
  const bytes = frames * channels * 2;
  text(0, "RIFF"); header.setUint32(4, 36 + bytes, true); text(8, "WAVE");
  text(12, "fmt "); header.setUint32(16, 16, true); header.setUint16(20, 1, true);           // PCM
  header.setUint16(22, channels, true); header.setUint32(24, rate, true);
  header.setUint32(28, rate * channels * 2, true); header.setUint16(32, channels * 2, true); header.setUint16(34, 16, true);
  text(36, "data"); header.setUint32(40, bytes, true);
  const data = new Int16Array(frames * channels);
  for (let c = 0; c < channels; c++) {
    const x = buffer.getChannelData(c);
    for (let i = 0; i < frames; i++) data[i * channels + c] = Math.max(-32768, Math.min(32767, Math.round(x[i] * 32767)));
  }
  return new Blob([header, data], { type: "audio/wav" });
}

class LiveSession {
  // One connection to the demo server, kept across seeks: restart() starts a fresh session on it from
  // another second of the recording (no new connection to set up, the place on the server kept).
  // pcm: Float32Array at sampleRate; from: the second of the recording the session starts at.
  // Callbacks: onReady(labelNames), onLabels(pointId, firstFrame, labels), onDone(), onIdle() (the server
  // closed the session after a pause), onError(message).
  constructor({ pcm, sampleRate, from, onReady, onLabels, onDone, onIdle, onError }) {
    this.sampleRate = sampleRate;
    this.closed = false;
    this.reset({ pcm, from, onReady, onLabels, onDone, onIdle, onError });
    this.connect();
  }

  reset({ pcm, from, ...callbacks }) {
    Object.assign(this, callbacks);
    this.pcm = pcm;
    this.next = Math.round(from * this.sampleRate);    // next sample to send
    this.ready = false;                                // until this session's "ready": older messages are ignored
    this.ended = false;
    this.finished = false;
    this.reconnected = false;
  }

  connect() {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${scheme}://${location.host}/api/live`);
    this.ws = ws;
    this.awaiting = 1;                                 // "ready"s still to come: the last one is this session's
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify({ type: "start" }));
    ws.onmessage = (event) => { if (ws === this.ws) this.receive(JSON.parse(event.data)); };
    ws.onclose = () => {
      if (ws !== this.ws || this.closed) return;
      if (!this.ready && !this.reconnected) {
        // A restart crossed the end of the old session (done or idle closed the connection): connect again.
        this.reconnected = true;
        this.connect();
      } else if (!this.finished) {
        this.onError("The connection to the demo server was lost.");
      }
    };
  }

  get usable() {
    // Can restart() continue on this connection (not closed, not ended by the server)?
    return !this.closed && !this.finished && this.ws.readyState <= WebSocket.OPEN;
  }

  restart({ pcm, from, ...callbacks }) {
    // A fresh session from `from` (a seek, Restart, another recording) on the same connection.
    this.reset({ pcm, from, ...callbacks });
    if (this.ws.readyState === WebSocket.OPEN) {
      this.awaiting += 1;
      this.ws.send(JSON.stringify({ type: "restart" }));
    }
    // still connecting: its "start" opens the session from the new position
  }

  receive(message) {
    if (this.closed) return;
    if (message.type === "ready") {
      this.awaiting -= 1;
      if (this.awaiting > 0) return;                   // a session already restarted again
      this.ready = true;
      this.onReady(message.labels);
    } else if (message.type === "error") {
      this.finished = true;
      this.onError(message.message);
    } else if (!this.ready) {
      // the end of the session before a restart: labels, done or idle that no longer count
    } else if (message.type === "labels") {
      for (const [point, value] of Object.entries(message.points)) this.onLabels(point, value.from, value.labels);
    } else if (message.type === "done") {
      this.finished = true;
      this.onDone();
    } else if (message.type === "idle") {
      this.finished = true;
      this.onIdle();
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
