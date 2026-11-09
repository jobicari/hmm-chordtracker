"""A Standard MIDI File reader, standard library only: the notes of every track, in seconds.

    read(path)              -> list[Track]: name, program, channel and notes (start s, end s, pitch, velocity)
    bass_notes(tracks)      the notes of every bass track (General MIDI bass program 32-39), merged; []
                            if the song has no bass part
    bass_line(notes)        (start s, pitch) of every onset where the lowest sounding note changes

Formats 0 and 1, ticks per quarter note (not SMPTE time). The tempo map (meta event 0x51) is read from
every track and applied to all of them, as format 1 requires. A note-on with velocity 0 is a note-off.
"""
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TEMPO = 500_000          # microseconds per quarter note (120 bpm) until the first tempo event
DRUM_CHANNEL = 9                 # channel 10 in General MIDI numbering
BASS_PROGRAMS = range(32, 40)    # General MIDI "Bass" family, 0-based


@dataclass
class Track:
    name: str = ""
    program: int | None = None
    channel: int | None = None
    notes: list[tuple[float, float, int, int]] = field(default_factory=list)


def _varlen(data: bytes, i: int) -> tuple[int, int]:
    """A variable-length quantity at data[i:]: (value, index after it)."""
    value = 0
    while True:
        b = data[i]
        i += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, i


def _events(data: bytes):
    """(absolute tick, kind, payload) for every event of one track chunk.

    kind: "note" (channel, pitch, velocity: 0 = off), "off" (channel, pitch), "program" (channel, program),
    "tempo" (microseconds per quarter), "name" (text).
    """
    i, tick, status = 0, 0, None
    while i < len(data):
        delta, i = _varlen(data, i)
        tick += delta
        b = data[i]
        if b == 0xFF:                                           # meta event
            kind = data[i + 1]
            length, i = _varlen(data, i + 2)
            body = data[i:i + length]
            i += length
            if kind == 0x51:
                yield tick, "tempo", int.from_bytes(body, "big")
            elif kind == 0x03:
                yield tick, "name", body.decode("latin-1")
            elif kind == 0x2F:
                return
            continue
        if b in (0xF0, 0xF7):                                   # sysex
            length, i = _varlen(data, i + 1)
            i += length
            continue
        if b & 0x80:
            status = b
            i += 1
        kind, channel = status & 0xF0, status & 0x0F           # running status: reuse the last one
        if kind in (0xC0, 0xD0):
            if kind == 0xC0:
                yield tick, "program", (channel, data[i])
            i += 1
            continue
        a, v = data[i], data[i + 1]
        i += 2
        if kind == 0x90:
            yield tick, "note", (channel, a, v)
        elif kind == 0x80:
            yield tick, "off", (channel, a)


def _chunks(raw: bytes):
    i = 0
    while i + 8 <= len(raw):
        kind, length = raw[i:i + 4], int.from_bytes(raw[i + 4:i + 8], "big")
        yield kind, raw[i + 8:i + 8 + length]
        i += 8 + length


class _Clock:
    """Ticks -> seconds through a tempo map [(tick, microseconds per quarter)]."""

    def __init__(self, tempi: list[tuple[int, int]], division: int):
        tempi = sorted(tempi) if tempi and tempi[0][0] == 0 else [(0, DEFAULT_TEMPO)] + sorted(tempi)
        self.ticks, self.tempi, self.seconds = [], [], []
        s, last_tick, last_tempo = 0.0, 0, tempi[0][1]
        for tick, tempo in tempi:
            s += (tick - last_tick) * last_tempo / 1e6 / division
            self.ticks.append(tick)
            self.tempi.append(tempo)
            self.seconds.append(s)
            last_tick, last_tempo = tick, tempo
        self.division = division

    def __call__(self, tick: int) -> float:
        k = bisect_right(self.ticks, tick) - 1
        return self.seconds[k] + (tick - self.ticks[k]) * self.tempi[k] / 1e6 / self.division


def read(path: Path | str) -> list[Track]:
    """Every track of a Standard MIDI File, notes in seconds, sorted by start."""
    chunks = list(_chunks(Path(path).read_bytes()))
    header = next(body for kind, body in chunks if kind == b"MThd")
    division = int.from_bytes(header[4:6], "big")
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    track_events = [list(_events(body)) for kind, body in chunks if kind == b"MTrk"]
    clock = _Clock([(t, p) for events in track_events for t, k, p in events if k == "tempo"], division)
    tracks = []
    for events in track_events:
        track, sounding = Track(), {}
        for tick, kind, payload in events:
            if kind == "name" and not track.name:
                track.name = payload
            elif kind == "program":
                track.channel, track.program = payload
            elif kind in ("note", "off"):
                channel, pitch = payload[0], payload[1]
                velocity = payload[2] if kind == "note" else 0
                if track.channel is None:
                    track.channel = channel
                key = (channel, pitch)
                if key in sounding:                             # a note ends (or is struck again)
                    start, vel = sounding.pop(key)
                    track.notes.append((clock(start), clock(tick), pitch, vel))
                if velocity:
                    sounding[key] = (tick, velocity)
        track.notes.sort()
        tracks.append(track)
    return tracks


def bass_notes(tracks: list[Track]) -> list[tuple[float, float, int, int]]:
    """The notes of every track with a General MIDI bass program (a song may switch between two basses),
    merged and sorted; drums excluded. Empty when no track plays a bass program."""
    notes = [n for t in tracks if t.notes and t.channel != DRUM_CHANNEL and t.program in BASS_PROGRAMS
             for n in t.notes]
    return sorted(notes)


def bass_line(notes: list[tuple[float, float, int, int]]) -> list[tuple[float, int]]:
    """(time, pitch) at every onset where the lowest note sounding changes pitch; a repeated note is one.

    Example: notes C2 0-1 s, C2 1-2 s, E2 2-3 s -> [(0.0, 36), (2.0, 40)].
    """
    out, active = [], []
    for start, end, pitch, _v in sorted(notes):
        active = [(e, p) for e, p in active if e > start] + [(end, pitch)]
        low = min(p for _, p in active)
        if not out or out[-1][1] != low:
            out.append((start, low))
    return out
