"""Tests for data_aam: AAM chord names and beat labels -> V5 targets."""
import numpy as np

from chordtracker import data_aam, data_v5

ARFF = """@RELATION 'Beat information of 0001'

@ATTRIBUTE 'Start time in seconds' NUMERIC
@ATTRIBUTE 'Bar count' NUMERIC
@ATTRIBUTE 'Quarter count' NUMERIC
@ATTRIBUTE 'Chord name' STRING

@DATA
0.0,1,1,'Fmaj'
0.5,1,2,'Fmaj'
1.0,1,3,'A#min'
1.6,1,4,'N.C.'
2.0,2,1,'BASS_NOTE_EXCEPTION'
"""


def test_harte_names():
    assert data_aam.harte("Fmaj") == "F:maj"
    assert data_aam.harte("A#min") == "A#:min"
    assert data_aam.harte("N.C.") == "N"
    assert data_aam.harte("BASS_NOTE_EXCEPTION") == "X"
    assert [data_v5.target(data_aam.harte(c)) for c in ("Fmaj", "A#min", "N.C.", "BASS_NOTE_EXCEPTION")] == [5, 22, -1, -1]


def test_one_segment_per_beat_and_the_last_lasts_the_median(tmp_path, monkeypatch):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "0001_beatinfo.arff").write_text(ARFF)
    monkeypatch.setattr(data_aam, "AAM_DIR", tmp_path)
    ann = data_aam.aam_annotation("0001")
    assert ann.labels == ["F:maj", "F:maj", "A#:min", "N", "X"]
    np.testing.assert_allclose(ann.intervals[:, 0], [0.0, 0.5, 1.0, 1.6, 2.0])
    np.testing.assert_allclose(ann.intervals[:, 1], [0.5, 1.0, 1.6, 2.0, 2.5])   # median beat 0.5 s


def test_the_last_beat_ends_at_the_end_mark(tmp_path, monkeypatch):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "0001_beatinfo.arff").write_text(ARFF)
    (tmp_path / "annotations" / "0001_segments.arff").write_text(
        "@DATA\n0.0,'A',92,'Fmaj','[Piano]','[ChordArpeggios]'\n2.3,'end',0,'','[]','[]'\n")
    monkeypatch.setattr(data_aam, "AAM_DIR", tmp_path)
    assert data_aam.aam_annotation("0001").intervals[-1].tolist() == [2.0, 2.3]


def test_targets_stop_after_the_last_beat(tmp_path, monkeypatch):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "0001_beatinfo.arff").write_text(ARFF)
    monkeypatch.setattr(data_aam, "AAM_DIR", tmp_path)
    y = data_v5.frame_targets(data_aam.aam_annotation("0001"), np.array([0.1, 0.7, 1.2, 1.8, 2.2, 3.0]))
    assert y.tolist() == [5, 5, 22, -1, -1, -1]


def test_song_ids_are_the_fetched_ones(tmp_path, monkeypatch):
    (tmp_path / "fetched.txt").write_text("0002\n0001\n0002\n")
    monkeypatch.setattr(data_aam, "AAM_DIR", tmp_path)
    assert data_aam.aam_song_ids() == ["0001", "0002"]
