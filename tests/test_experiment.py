"""Tests for chordtracker.experiment (component 6).

Unit tests use hand-made posteriors and a fake extractor/emission model, so they run in a moment
and need no audio. The test marked `slow` runs the real functions on one GuitarSet train take
(no val or test take is touched): run it with `pytest -m slow`.
"""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from chordtracker import experiment
from chordtracker.data.data import ChordAnnotation

CFG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "v0.yaml").read_text())
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "processed" / "guitarset" / "manifest.csv"
LABELS = ["C:maj", "G:maj", "N"]


def evidence(posteriors, reference_segments, song_id="take"):
    """TakeEvidence with frames every 1 s (frame t at t s) and the given ((start, end, label), ...) reference."""
    posteriors = np.asarray(posteriors, dtype=float)
    intervals = np.array([[start, end] for start, end, _ in reference_segments], dtype=float)
    reference = ChordAnnotation(intervals, [label for _, _, label in reference_segments])
    return experiment.TakeEvidence(song_id, posteriors, np.arange(posteriors.shape[1], dtype=float),
                                   reference, float(intervals[-1, 1]))


def blip_take():
    """10 frames of C:maj with a G:maj blip at frame 5; reference C:maj 0-10 s.

    Framewise follows the blip (1 s wrong, 2 changes), Viterbi at 0.9 holds C:maj (all right, 0 changes).
    """
    posteriors = np.tile([[0.6], [0.3], [0.1]], (1, 10))
    posteriors[:, 5] = [0.3, 0.6, 0.1]
    return evidence(posteriors, [(0, 10, "C:maj")])


# ------------------------------------------------------------------------ evidence

def test_take_evidence_reads_the_record_paths(tmp_path):
    """record.audio goes to the extractor and record.lab is read as is; posteriors and frame times come from the given models."""
    (tmp_path / "labs").mkdir()
    (tmp_path / "labs" / "t.lab").write_text("0.0\t2.0\tC:maj\n2.0\t3.0\tG:maj\n")
    seen = {}

    class FakeExtractor:
        def extract(self, path):
            seen["audio"] = path
            return np.ones((12, 4))

        def frame_times(self, n_frames):
            return np.arange(n_frames) * 0.5

    class FakeEmission:
        def posteriors(self, chroma):
            return np.full((3, chroma.shape[1]), 1 / 3)

    record = SimpleNamespace(song_id="t", group="g", audio=tmp_path / "audio" / "t.wav", lab=tmp_path / "labs" / "t.lab")
    result = experiment.take_evidence(record, FakeExtractor(), FakeEmission())
    assert seen["audio"] == tmp_path / "audio" / "t.wav"
    assert result.song_id == "t" and result.posteriors.shape == (3, 4) and result.duration == 3.0
    np.testing.assert_allclose(result.frame_times, [0.0, 0.5, 1.0, 1.5])
    assert result.reference.labels == ["C:maj", "G:maj"]


# ------------------------------------------------------------------------ scoring

def test_score_path_returns_wcsr_and_changes():
    """Path C,C,G,G on reference C 0-2, G 2-4: boundary at 1.5 s, so 0.5 s wrong -> WCSR 3.5 / 4, 1 change in 4 s = 15/min."""
    take = evidence(np.zeros((3, 4)), [(0, 2, "C:maj"), (2, 4, "G:maj")])
    song_score, changes = experiment.score_path(np.array([0, 0, 1, 1]), LABELS, take)
    assert song_score.wcsr == pytest.approx(3.5 / 4)
    assert changes == pytest.approx(15.0)


def test_sweep_has_one_summary_row_per_system_and_one_per_song_row_per_take():
    """Grid [0.5, 0.9] on 2 takes -> 3 summary rows (framewise, 0.5, 0.9) and 3 x 2 per-song rows."""
    summary_rows, per_song_rows = experiment.sweep_p_self([blip_take(), blip_take()], LABELS, [0.5, 0.9])
    assert [row["system"] for row in summary_rows] == ["framewise", 0.5, 0.9]
    assert len(per_song_rows) == 6 and all(row["n_takes"] == 2 for row in summary_rows)


def test_sweep_viterbi_smooths_the_blip_that_framewise_follows():
    """On the blip take framewise scores 9 / 10 with 12 changes/min; Viterbi 0.9 scores 10 / 10 with 0."""
    summary_rows, _ = experiment.sweep_p_self([blip_take()], LABELS, [0.9])
    framewise, viterbi = summary_rows
    assert framewise["wcsr"] == pytest.approx(0.9) and framewise["mean_changes_per_min"] == pytest.approx(12.0)
    assert viterbi["wcsr"] == 1.0 and viterbi["mean_changes_per_min"] == 0.0


def test_sweep_uniform_p_self_matches_framewise():
    """p_self = 1/3 on 3 states makes every transition equal, so its row equals the framewise row."""
    rng = np.random.default_rng(0)
    posteriors = rng.random((3, 50))
    take = evidence(posteriors / posteriors.sum(axis=0), [(0, 25, "C:maj"), (25, 50, "G:maj")])
    framewise, uniform = experiment.sweep_p_self([take], LABELS, [1 / 3])[0]
    assert framewise["wcsr"] == uniform["wcsr"]
    assert framewise["mean_changes_per_min"] == uniform["mean_changes_per_min"]


def test_choose_p_self_takes_the_best_and_breaks_ties_towards_less_smoothing():
    """0.9 -> 0.61, 0.99 -> 0.63, 0.999 -> 0.63: 0.99 wins; framewise (0.70) is never chosen; no Viterbi rows raises."""
    rows = [{"system": "framewise", "wcsr": 0.70}, {"system": 0.9, "wcsr": 0.61},
            {"system": 0.99, "wcsr": 0.63}, {"system": 0.999, "wcsr": 0.63}]
    assert experiment.choose_p_self(rows) == 0.99
    with pytest.raises(ValueError):
        experiment.choose_p_self(rows[:1])


# ------------------------------------------------------------------------ output

def test_write_results_round_trip(tmp_path):
    """summary.json reads back as the same dict, per_song.csv has a header and one line per row."""
    summary_rows, per_song_rows = experiment.sweep_p_self([blip_take()], LABELS, [0.9])
    summary = {"version": "v0", "split": "val", "rows": summary_rows, "chosen_p_self": 0.9}
    experiment.write_results(summary, per_song_rows, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "summary.json").read_text()) == summary
    with open(tmp_path / "out" / "per_song.csv") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2 and rows[1]["system"] == "0.9"


# ------------------------------------------------------------------------ real data

@pytest.mark.slow
@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifest not found in data/processed/guitarset")
def test_real_train_take_runs_through_the_sweep():
    """On train take 00_BN1-129-Eb_comp: evidence loads, every system scores in [0, 1], 1/25 equals framewise."""
    from chordtracker import emission
    from chordtracker.data.data import load_manifest
    from chordtracker.features import ChromaCQTExtractor

    record = next(r for r in load_manifest(MANIFEST) if r.song_id == "00_BN1-129-Eb_comp")
    model = emission.TemplateEmission.from_config(CFG)
    take = experiment.take_evidence(record, ChromaCQTExtractor.from_config(CFG), model)
    assert take.posteriors.shape == (25, len(take.frame_times))

    summary_rows, _ = experiment.sweep_p_self([take], model.labels, [1 / 25, 0.99])
    framewise, uniform, sticky = summary_rows
    assert all(0.0 <= row["wcsr"] <= 1.0 for row in summary_rows)
    assert uniform["wcsr"] == framewise["wcsr"]
    assert sticky["mean_changes_per_min"] < framewise["mean_changes_per_min"]
