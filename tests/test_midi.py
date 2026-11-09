"""Behaviour of the MIDI reader: notes in seconds through the tempo map, running status, note-on velocity 0
as note-off, the bass tracks merged, and the bass line's changes of lowest note."""
import struct

from chordtracker import midi


def varlen(n: int) -> bytes:
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append(0x80 | (n & 0x7F))
        n >>= 7
    return bytes(reversed(out))


def track(events: list[tuple[int, bytes]]) -> bytes:
    body = b"".join(varlen(d) + e for d, e in events) + b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(body)) + body


def smf(*tracks: bytes, division: int = 480) -> bytes:
    return b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), division) + b"".join(tracks)


def write(tmp_path, data):
    path = tmp_path / "x.mid"
    path.write_bytes(data)
    return path


def test_notes_follow_the_tempo_map(tmp_path):
    """120 bpm for the first quarter, then 60 bpm: a note on beats 2-3 lasts 1 s and starts at 0.5 s."""
    conductor = track([(0, b"\xff\x51\x03" + (500_000).to_bytes(3, "big")),
                       (480, b"\xff\x51\x03" + (1_000_000).to_bytes(3, "big"))])
    bass = track([(0, b"\xff\x03\x04BASS"), (0, b"\xc1\x21"), (480, b"\x91\x24\x64"), (480, b"\x81\x24\x00")])
    tracks = midi.read(write(tmp_path, smf(conductor, bass)))
    assert tracks[1].name == "BASS" and tracks[1].program == 33 and tracks[1].channel == 1
    assert tracks[1].notes == [(0.5, 1.5, 36, 100)]


def test_running_status_and_velocity_zero_end_notes(tmp_path):
    t = track([(0, b"\x90\x3c\x50"), (480, b"\x3c\x00"), (0, b"\x40\x50"), (240, b"\x40\x00")])
    notes = midi.read(write(tmp_path, smf(t)))[0].notes                      # default 120 bpm
    assert notes == [(0.0, 0.5, 60, 80), (0.5, 0.75, 64, 80)]


def test_bass_notes_merge_every_bass_track_and_skip_others(tmp_path):
    a = track([(0, b"\xc0\x20"), (0, b"\x90\x24\x50"), (480, b"\x80\x24\x00")])           # finger bass
    b = track([(0, b"\xc1\x27"), (960, b"\x91\x28\x50"), (480, b"\x81\x28\x00")])         # synth bass
    piano = track([(0, b"\xc2\x00"), (0, b"\x92\x18\x50"), (480, b"\x82\x18\x00")])
    drums = track([(0, b"\xc9\x20"), (0, b"\x99\x23\x50"), (480, b"\x89\x23\x00")])
    notes = midi.bass_notes(midi.read(write(tmp_path, smf(a, b, piano, drums))))
    assert [n[2] for n in notes] == [36, 40]
    assert midi.bass_notes(midi.read(write(tmp_path, smf(piano)))) == []


def test_the_bass_line_lists_changes_of_the_lowest_note():
    notes = [(0.0, 1.0, 36, 80), (1.0, 2.0, 36, 80), (2.0, 3.0, 40, 80), (2.5, 2.8, 52, 80), (3.0, 4.0, 43, 80)]
    assert midi.bass_line(notes) == [(0.0, 36), (2.0, 40), (3.0, 43)]
