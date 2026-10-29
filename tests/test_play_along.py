"""Behaviour of the listening prototype's logic: chord names, the annotation carousel, the system's labels,
what the window shows when, and the delay that keeps every label on time."""

import numpy as np
import pytest
import torch

from chordtracker import model_v6, play_along as pa
from chordtracker.learned_emission import LABELS

SR, HOP = 22050, 2048


@pytest.mark.parametrize("label,name", [("E:min7", "Em7"), ("A", "A"), ("G/3", "G"), ("N", "N.C."),
                                        ("C:maj", "C"), ("F#:min", "F#m"), ("D:7(b9)", "D7"), ("Bb:sus4/5", "Bbsus4")])
def test_chord_names(label, name):
    assert pa.chord_name(label) == name


def test_the_reference_carousel_shows_the_chord_now_and_the_next_different_one(tmp_path):
    lab = tmp_path / "song.lab"
    lab.write_text("0.0 1.0 N\n1.0 2.0 E:min\n2.0 3.0 E:min\n3.0 4.0 A\n")
    ref = pa.Reference.from_lab(lab)
    assert ref.at(0.5) == ("N.C.", "Em")
    assert ref.at(1.5) == ("Em", "A")
    assert ref.at(2.5) == ("Em", "A")                    # a repeated label is not a change
    assert ref.at(3.5) == ("A", "")
    assert ref.at(-1.0) == ("", "N.C.")


def test_the_song_list_has_every_annotation_by_album(tmp_path):
    for album, song in (("02_-_With_the_Beatles", "14_-_Money"), ("01_-_Please_Please_Me", "01_-_I_Saw_Her")):
        (tmp_path / album).mkdir()
        (tmp_path / album / f"{song}.lab").write_text("0 1 N\n")
    names = [n for n, _ in pa.song_list(tmp_path)]
    assert names == ["01 - Please Please Me | 01 - I Saw Her", "02 - With the Beatles | 14 - Money"]


def test_the_onset_is_the_first_block_out_of_silence():
    onset = pa.Onset(threshold=1e-3)
    onset.update(np.zeros(2048), 0.0)
    assert onset.time is None
    onset.update(np.full(2048, 0.01), 0.093)
    onset.update(np.full(2048, 0.5), 0.186)
    assert onset.time == 0.093


def test_the_window_shows_the_chords_lead_seconds_ahead():
    labels = [0] * 10 + [7] * 10                          # C:maj for frames 0..9, G:maj from frame 10
    hop_s = HOP / SR
    at = lambda t, lead: pa.shown(labels, None, None, t, lead, 0.0, SR, HOP)
    assert at(9.5 * hop_s, 0.0)["system"] == "C" and at(9.5 * hop_s, 0.0)["system_next"] == "G"
    assert at(9.5 * hop_s, hop_s)["system"] == "G"        # one frame of lead: the next chord shows a frame early
    assert at(25 * hop_s, 0.0) == {**at(25 * hop_s, 0.0), "system": "", "ready": False}


def test_the_reference_starts_at_the_onset_and_can_be_nudged(tmp_path):
    lab = tmp_path / "s.lab"
    lab.write_text("0.0 1.0 C\n1.0 2.0 G\n")
    ref = pa.Reference.from_lab(lab)
    show = lambda t, nudge: pa.shown([], ref, 5.0, t, 0.0, nudge, SR, HOP)["reference"]
    assert show(5.5, 0.0) == "C" and show(6.5, 0.0) == "G"
    assert show(5.95, 0.1) == "G"


def test_the_required_delay_covers_the_total_lookahead_and_the_lead():
    assert pa.total_lookahead("0.28 s") == 3 and pa.total_lookahead("0.65 s") == 7
    assert pa.required_delay("0.28 s", 0.0, SR, HOP) == pytest.approx(4 * HOP / SR)
    assert pa.required_delay("0.65 s", 0.2, SR, HOP) == pytest.approx(8 * HOP / SR + 0.2)


def test_the_analysis_labels_every_frame_at_every_point():
    """Blocks of a short clip in, one label per frame out for each point once the song is flushed in time."""
    torch.manual_seed(0)
    model = model_v6.ChordCRNN(bidirectional=False)
    cfg = {"audio": {"sr": SR, "mono": True, "harmonic_margin": 4.0, "harmonic_median_frames": 9},
           "cqt": {"hop_length": HOP, "fmin": "F#1", "bins_per_octave": 24, "n_bins": 168, "tuning_seconds": 10.0},
           "compression": {"enabled": True, "level_percentile": 99, "gamma": 10}}
    Q = np.full((24, 24), 1 / 23)
    np.fill_diagonal(Q, 0)
    a = pa.Analysis(cfg, model, np.full(24, 1 / 24), Q)
    y = (0.1 * np.sin(2 * np.pi * 220 * np.arange(SR * 3) / SR)).astype(np.float32)
    for i in range(0, len(y), 1000):                      # any block size
        a.step(y[i:i + 1000])
    n = len(a.labels["0.28 s"])
    assert n > 20
    assert len(a.labels["0.46 s"]) == n - 2 and len(a.labels["0.65 s"]) == n - 4
    assert all(0 <= l < len(LABELS) for l in a.labels["0.28 s"])
