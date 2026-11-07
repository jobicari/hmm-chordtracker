"""Behaviour of the ChoCo export loader: the split picks the takes, each keeps its labels and tuning, the
audio check catches a changed file, and the tuned CQT puts a detuned note back in its A440 bin."""
import csv
import hashlib
import json

import jams
import librosa
import numpy as np
import pytest
import soundfile as sf

from chordtracker import data_choco

SR, HOP = 22050, 2048
CQT_CFG = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0},
           "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
           "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}
FIELDS = ["id", "partition", "status", "labels_from", "jams", "audio", "title", "artist", "duration",
          "tuning_cents", "youtube_id", "verdict", "sha256"]


def write_recording(folder, rec, chords, cents, seconds=3.0):
    """One recording as the export writes it: WAV, JAMS chord annotation, manifest row."""
    y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * SR)) / SR)).astype(np.float32)
    sf.write(folder / f"{rec}.wav", y, SR)
    jam = jams.JAMS()
    jam.file_metadata.duration = seconds
    ann = jams.Annotation(namespace="chord")
    for start, dur, value in chords:
        ann.append(time=start, duration=dur, value=value)
    jam.annotations.append(ann)
    jam.save(str(folder / f"{rec}.jams"))
    sha = hashlib.sha256((folder / f"{rec}.wav").read_bytes()).hexdigest()
    return {"id": rec, "partition": folder.name, "status": "pass", "labels_from": "placed", "jams": f"{rec}.jams",
            "audio": f"{rec}.wav", "title": rec, "artist": "a", "duration": seconds, "tuning_cents": cents,
            "youtube_id": "x", "verdict": "pass", "sha256": sha}


@pytest.fixture
def export(tmp_path):
    """A tiny export: two isophonics recordings and one jaah, plus a split file."""
    for partition in data_choco.PARTITIONS:
        (tmp_path / partition).mkdir()
    rows = {"isophonics": [write_recording(tmp_path / "isophonics", "isophonics_0", [(0, 1.5, "A"), (1.5, 1.5, "E:7")], 32),
                           write_recording(tmp_path / "isophonics", "isophonics_1", [(0, 3, "D:min7")], -42)],
            "jaah": [write_recording(tmp_path / "jaah", "jaah_3", [(0, 3, "Bb:maj7")], 12)]}
    for partition in data_choco.PARTITIONS:
        with open(tmp_path / partition / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows.get(partition, []))
    split = tmp_path / "choco.json"
    split.write_text(json.dumps({"seed": 0, "train": ["isophonics_0", "jaah_3"], "val": ["isophonics_1"], "test": []}))
    return {"data": {"choco_export": str(tmp_path)}}, split


def test_the_split_picks_the_takes_with_their_labels_and_tuning(export):
    cfg, split = export
    train = data_choco.choco_takes("train", cfg, split)
    assert [t.key for t in train] == ["choco/isophonics_0", "choco/jaah_3"]
    assert train[0].annotation.labels == ["A", "E:7"]
    np.testing.assert_allclose(train[0].annotation.intervals, [[0, 1.5], [1.5, 3.0]])
    assert [t.tuning_cents for t in train] == [32.0, 12.0]
    assert data_choco.choco_takes("val", cfg, split)[0].tuning_cents == -42.0


def test_test_is_loaded_only_by_the_frozen_run(export):
    cfg, split = export
    with pytest.raises(ValueError):
        data_choco.choco_takes("test", cfg, split)
    assert data_choco.choco_takes("test", cfg, split, frozen=True) == []


def test_the_audio_check_catches_a_changed_or_missing_file(export):
    cfg, split = export
    take = data_choco.choco_takes("train", cfg, split)[0]
    data_choco.check_audio(take)
    y, _ = sf.read(take.audio)
    sf.write(take.audio, 0.5 * y, SR)
    with pytest.raises(ValueError):
        data_choco.check_audio(take)
    take.audio.unlink()
    with pytest.raises(FileNotFoundError):
        data_choco.check_audio(take)


def test_cents_to_bins():
    assert data_choco.tuning_bins(50, 24) == 1.0
    assert data_choco.tuning_bins(-42, 24) == pytest.approx(-0.84)
    assert data_choco.tuning_bins(42, 12) == pytest.approx(0.42)


@pytest.mark.parametrize("cents", [-42, 30])
def test_the_tuned_cqt_puts_a_detuned_note_in_its_a440_bin(cents):
    """A4 played `cents` off A440: at its tuning it sits on the A4 bin, its two neighbours equal; at A440 it
    leans towards the neighbour on the side it is off to."""
    t = np.arange(4 * SR) / SR
    y = (0.3 * np.sin(2 * np.pi * 440 * 2 ** (cents / 1200) * t)).astype(np.float32)
    a4 = 2 * (librosa.note_to_midi("A4") - librosa.note_to_midi("F#1"))      # 2 bins per semitone from F#1
    toward = 1 if cents > 0 else -1

    def around_a4(c):
        C = data_choco.tuned_log_cqt_of_audio(y, CQT_CFG, c)[:, 20:-5].mean(axis=1)
        return C[a4 - toward], C[a4], C[a4 + toward]

    away, centre, near = around_a4(cents)
    assert centre > max(away, near)
    assert abs(near - away) < 0.05 * centre
    away, centre, near = around_a4(0)
    assert near - away > 0.2 * centre
