"""Listening prototype: play a recording with delayed playback and show the chords live.

    python scripts/play_along.py --list-devices
    python scripts/play_along.py --input-device "VB-Cable A"        # e.g. Spotify's output set to VB-Cable A
    python scripts/play_along.py --file song.wav                     # a file, read at real-time pace

The recording is analysed block by block with V7.3's live chain (C2 grid, input look-ahead 2, results/v7_2/
c2_la2_seed0) and played back on the output device `delay` seconds late, so every chord is ready when its
moment is heard. The window shows the system's chord now and next, and the Isophonics annotation of the song
picked in the drop-down (now and next). Pick the song first, then start it in Spotify: the annotation's
clock starts when the input rises out of silence.

Keys: 1 / 2 / 3 operating point (0.28 / 0.46 / 0.65 s of total look-ahead); Up / Down extra delay
+-50 ms; Right / Left lead +-50 ms (chords shown earlier); ] / [ nudge the annotation +-50 ms; space
reset for the next song ([ shows the annotation earlier, ] later). Every change is logged to data/play_along/<start time>.json.
"""
import argparse
import json
import multiprocessing as mp
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import librosa
import numpy as np
import sounddevice as sd

from chordtracker import paths, play_along as pa

sys.path.insert(0, str(Path(__file__).parent))
import run_v5  # noqa: E402
import run_v7_2  # noqa: E402

RUN = run_v7_2.run_dir("C2", pa.INPUT_LOOKAHEAD, 0)
BEATLES = paths.DATA_DIR / "beatles" / "chordlab"
LOG_DIR = paths.DATA_DIR / "play_along"
STEP = 0.05
BG, TEXT, MUTED, LINE, ACCENT, WARN = "#1c1c1e", "#f2f2f2", "#a8a8ad", "#3a3a3c", "#5eb0ff", "#ff6b6b"


class Engine:
    """Capture and delayed playback in the audio callbacks; the analysis in its own process (analysis_worker).

    Input time = input samples since the last reset / rate. The callbacks only copy samples, so the
    analysis (and Python's interpreter lock) can never make the playback stutter.
    """

    def __init__(self, rate: int, cfg: dict, Q):
        self.rate, self.cfg = rate, cfg
        self.sr, self.hop = cfg["audio"]["sr"], cfg["cqt"]["hop_length"]
        self.ring = np.zeros((rate * 30, 2), dtype=np.float32)       # the last 30 s of input
        self.written = 0                                             # input samples since start (device rate)
        self.played = 0                                              # next output sample (input clock)
        self.delay_s, self.reposition = 0.5, True
        self.level_db = -200.0                                       # input level of the last block, dBFS
        self.xruns = 0                                               # input overflows + output underflows
        self.jumps = 0                                               # playback repositions after drift
        ctx = mp.get_context("spawn")
        self.blocks, self.results = ctx.Queue(), ctx.Queue()
        self.worker = ctx.Process(target=pa.analysis_worker, daemon=True,
                                  args=(self.blocks, self.results, cfg, str(RUN), Q, rate))
        self.worker.start()
        self.generation = 0
        self.reset()

    def reset(self) -> None:
        """A new song: analysis, onset and input time start again from the next sample."""
        self.generation += 1
        self.base = self.written
        self.labels = {p: [] for p in pa.OPERATING_POINTS}
        self.onset_time = None
        self.blocks.put(("reset", self.generation))

    def set_delay(self, delay_s: float) -> None:
        self.delay_s, self.reposition = delay_s, True

    def collect(self) -> None:
        """Take the analysis results that have arrived (called from the window's timer)."""
        while True:
            try:
                generation, new, onset = self.results.get_nowait()
            except queue.Empty:
                return
            if generation == self.generation:
                for p, labels in new.items():
                    self.labels[p].extend(labels)
                self.onset_time = onset

    # ------------------------------------------------------------ audio callbacks
    def push(self, block: np.ndarray, status=None) -> None:
        """Input callback: store the block for playback and send its mono mix to the analysis."""
        if status:
            self.xruns += 1
        block = np.asarray(block, dtype=np.float32)
        if block.shape[1] == 1:
            block = np.repeat(block, 2, axis=1)
        n = len(block)
        start = self.written % len(self.ring)
        first = min(n, len(self.ring) - start)
        self.ring[start:start + first] = block[:first, :2]
        self.ring[:n - first] = block[first:, :2]
        self.written += n
        self.blocks.put((self.generation, block.mean(axis=1)))
        self.level_db = 20 * np.log10(max(float(np.sqrt(np.mean(block ** 2))), 1e-10))

    def pull(self, out: np.ndarray, frames: int, status=None) -> None:
        """Output callback: the input `delay_s` seconds ago, played continuously.

        The read position moves only when the delay is changed, or if the two devices' clocks have drifted
        more than 0.25 s apart; otherwise samples are played in order, without gaps or repeats.
        """
        if status:
            self.xruns += 1
        target = self.written - int(self.delay_s * self.rate)
        if self.reposition or abs(self.played - target) > self.rate // 4:
            self.jumps += 0 if self.reposition else 1
            self.played, self.reposition = target, False
        pos = self.played + np.arange(frames)
        ok = (pos >= 0) & (pos < self.written) & (pos > self.written - len(self.ring))
        out[:] = 0
        out[ok] = self.ring[pos[ok] % len(self.ring)]
        self.played += frames

    def playback_s(self) -> float:
        """Input time (since the last reset) of the sample being heard now: the input clock minus the delay."""
        return (self.written - self.base) / self.rate - self.delay_s


class Window:
    def __init__(self, engine: Engine, log_file: Path):
        self.engine, self.log_file = engine, log_file
        self.point, self.extra, self.lead, self.nudge = "0.28 s", 0.0, 0.0, 0.0
        self.reference, self.song = None, None
        self.log = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "events": []}
        self.root = tk.Tk()
        self.root.title("Chord tracker: play along")
        self.root.configure(bg=BG)
        self.songs = dict(pa.song_list(BEATLES))
        top = tk.Frame(self.root, bg=BG, padx=12, pady=10)
        top.pack(fill="x")
        tk.Label(top, text="Song", font=("Helvetica", 15), fg=TEXT, bg=BG).pack(side="left")
        self.pick = ttk.Combobox(top, values=list(self.songs), width=70, state="readonly")
        self.pick.pack(side="left", padx=8)
        self.pick.bind("<<ComboboxSelected>>", self.choose)
        big = ("Helvetica", 96, "bold")
        self.system = tk.Label(self.root, text="", font=big, fg=TEXT, bg=BG)
        self.system.pack(pady=(10, 0))
        self.system_next = tk.Label(self.root, text="", font=("Helvetica", 30), fg=MUTED, bg=BG)
        self.system_next.pack()
        tk.Frame(self.root, height=2, bg=LINE).pack(fill="x", padx=20, pady=12)
        tk.Label(self.root, text="ISOPHONICS", font=("Helvetica", 15, "bold"), fg=MUTED, bg=BG).pack()
        row = tk.Frame(self.root, bg=BG)
        row.pack()
        self.reference_now = tk.Label(row, text="", font=("Helvetica", 52, "bold"), fg=ACCENT, bg=BG)
        self.reference_now.pack(side="left", padx=20)
        self.reference_next = tk.Label(row, text="", font=("Helvetica", 36), fg=MUTED, bg=BG)
        self.reference_next.pack(side="left", padx=20)
        tk.Frame(self.root, height=2, bg=LINE).pack(fill="x", padx=20, pady=12)
        panel = tk.Frame(self.root, bg=BG)
        panel.pack(padx=20, pady=(0, 16))
        self.fields = {}
        rows = [("Look-ahead", "1 / 2 / 3"), ("Playback delay", "up / down: extra"), ("Lead", "left / right"),
                ("Annotation nudge", "[ earlier / ] later"), ("Input", "space: reset"), ("Analysis", "")]
        for r, (name, keys_hint) in enumerate(rows):
            tk.Label(panel, text=name, font=("Helvetica", 17), fg=MUTED, bg=BG, anchor="e").grid(
                row=r, column=0, sticky="e", padx=(0, 14), pady=2)
            self.fields[name] = tk.Label(panel, text="", font=("Menlo", 18, "bold"), fg=TEXT, bg=BG, anchor="w")
            self.fields[name].grid(row=r, column=1, sticky="w")
            tk.Label(panel, text=keys_hint, font=("Helvetica", 15), fg=MUTED, bg=BG, anchor="w").grid(
                row=r, column=2, sticky="w", padx=(18, 0))
        keys = {"1": lambda: self.set_point("0.28 s"), "2": lambda: self.set_point("0.46 s"),
                "3": lambda: self.set_point("0.65 s"), "<Up>": lambda: self.change("extra", STEP),
                "<Down>": lambda: self.change("extra", -STEP), "<Right>": lambda: self.change("lead", STEP),
                "<Left>": lambda: self.change("lead", -STEP), "]": lambda: self.change("nudge", STEP),
                "[": lambda: self.change("nudge", -STEP), "<space>": self.reset}
        for key, action in keys.items():
            self.root.bind(key, lambda _e, a=action: a())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.apply()
        self.tick()

    def record(self, what: str) -> None:
        self.log["events"].append({"time": time.strftime("%H:%M:%S"), "event": what, "song": self.song,
                                   "point": self.point, "extra_delay_s": round(self.extra, 3),
                                   "lead_s": round(self.lead, 3), "nudge_s": round(self.nudge, 3),
                                   "delay_s": round(self.engine.delay_s, 3)})
        self.log_file.write_text(json.dumps(self.log, indent=2))

    def apply(self) -> None:
        e = self.engine
        e.set_delay(pa.required_delay(self.point, self.lead, e.sr, e.hop) + self.extra)

    def set_point(self, point: str) -> None:
        self.point = point
        self.apply()
        self.record("point")

    def change(self, what: str, step: float) -> None:
        value = getattr(self, what) + step
        if what in ("extra", "lead"):
            value = max(0.0, value)
        setattr(self, what, value)
        self.apply()
        self.record(what)

    def choose(self, _event=None) -> None:
        self.song = self.pick.get()
        self.reference = pa.Reference.from_lab(self.songs[self.song])
        self.root.focus_set()                      # keys go to the window again, not to the drop-down
        self.reset()

    def reset(self) -> None:
        self.engine.reset()
        self.record("reset")

    def tick(self) -> None:
        e = self.engine
        e.collect()
        labels = e.labels[self.point]
        onset = e.onset_time
        s = pa.shown(labels, self.reference, onset, e.playback_s(), self.lead, self.nudge, e.sr, e.hop)
        self.system.config(text=s["system"] or "...")
        self.system_next.config(text=f"next: {s['system_next']}" if s["system_next"] else "")
        if self.reference is None:
            self.reference_now.config(text="pick a song")
            self.reference_next.config(text="")
        else:
            self.reference_now.config(text=s["reference"] or ("waiting for the music" if onset is None else ""))
            self.reference_next.config(text=f"next: {s['reference_next']}" if s["reference_next"] else "")
        lookahead = pa.total_lookahead(self.point) * e.hop / e.sr
        f = self.fields
        f["Look-ahead"].config(text=f"{self.point}  ({lookahead:.2f} s)")
        f["Playback delay"].config(text=f"{e.delay_s:.2f} s   (extra {self.extra:.2f} s)")
        f["Lead"].config(text=f"{self.lead:.2f} s")
        f["Annotation nudge"].config(text=f"{self.nudge:+.2f} s")
        silent = e.level_db < -100
        f["Input"].config(text=f"{e.level_db:.0f} dBFS" + ("   no signal" if silent else ""), fg=WARN if silent else TEXT)
        f["Analysis"].config(text=f"onset {'-' if onset is None else f'{onset:.1f} s'}   frames {len(labels)}   "
                                  f"dropouts {e.xruns}   resyncs {e.jumps}")
        self.root.after(30, self.tick)

    def close(self) -> None:
        self.record("close")
        self.root.destroy()


def feed_file(path: Path, engine: Engine) -> None:
    """Read a file at real-time pace into the engine, as an input device would."""
    y, _ = librosa.load(str(path), sr=engine.rate, mono=False)
    y = np.atleast_2d(y).T
    block, start = 1024, time.perf_counter()
    for i in range(0, len(y), block):
        while time.perf_counter() - start < i / engine.rate:
            time.sleep(0.002)
        engine.push(y[i:i + block])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--input-device", help="name (or part of it) of the input, e.g. 'VB-Cable A'")
    parser.add_argument("--file", type=Path, help="play a file instead of an input device")
    parser.add_argument("--output-device", help="name (or part of it) of the output; default: the system's")
    args = parser.parse_args(argv)
    if args.list_devices:
        print(sd.query_devices())
        return
    if (args.input_device is None) == (args.file is None):
        parser.error("give exactly one of --input-device and --file")

    base = run_v5.load_config(paths.REPO_ROOT / "configs" / "v7_2.yaml")
    cfg = run_v7_2.cell_config(base, "C2", pa.INPUT_LOOKAHEAD, 0)
    rate = int(sd.query_devices(args.output_device, "output")["default_samplerate"])
    engine = Engine(rate, cfg, run_v5.change_prior(cfg))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    window = Window(engine, LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json")

    out = sd.OutputStream(device=args.output_device, samplerate=rate, channels=2, dtype="float32",
                          latency="high", callback=lambda o, f, t, s: engine.pull(o, f, s))
    if args.file:
        threading.Thread(target=feed_file, args=(args.file, engine), daemon=True).start()
        with out:
            window.root.mainloop()
    else:
        channels = min(2, int(sd.query_devices(args.input_device, "input")["max_input_channels"]))
        inp = sd.InputStream(device=args.input_device, samplerate=rate, channels=channels, dtype="float32",
                             latency="low", callback=lambda i, f, t, s: engine.push(i, s))   # VB-Cable zero-fills large input buffers
        with inp, out:
            window.root.mainloop()


if __name__ == "__main__":
    main()
