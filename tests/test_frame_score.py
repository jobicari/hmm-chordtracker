"""frame_score's tables give evaluate.score's numbers for any frame path."""
import numpy as np
import pytest

from chordtracker import evaluate, frame_score
from chordtracker.data.data import ChordAnnotation

LABELS = ["C:maj", "G:maj", "A:min", "G:7", "C:maj7", "N"]


def reference(end: float) -> ChordAnnotation:
    edges = [0.0, 0.4, 1.13, 1.5, 2.2, 2.9, end]
    labels = ["N", "C:maj", "G:7", "A:min7", "B:hdim7", "C:maj7/3"]
    return ChordAnnotation(np.array(list(zip(edges[:-1], edges[1:]))), labels)


def test_cells_cover_the_song_once():
    c = frame_score.cells(3, 0.1, 0.3)
    assert np.allclose(c, [[0, 0.05], [0.05, 0.15], [0.15, 0.3]])


def test_docstring_example():
    ann = ChordAnnotation(np.array([[0.0, 0.3]]), ["C:maj"])
    M, S = frame_score.tables(ann, ["C:maj", "G:maj"], 3, 0.1, "majmin")
    assert frame_score.score_path(M, S, np.array([0, 0, 1])) == pytest.approx((0.15, 0.3))


@pytest.mark.parametrize("metric", ["majmin", "sevenths"])
@pytest.mark.parametrize("n_frames, end", [(40, 3.7), (40, 3.2), (36, 3.7)])   # frames past the end, and short
def test_tables_equal_evaluate_score(metric, n_frames, end):
    rng = np.random.default_rng(n_frames)
    ann = reference(end)
    period = 0.0929
    M, S = frame_score.tables(ann, LABELS, n_frames, period, metric)
    times = np.arange(n_frames) * period
    for _ in range(20):
        path = rng.integers(0, len(LABELS), n_frames)
        path[5:12] = path[5]                                          # runs, as decoders give
        s = evaluate.score(ann, evaluate.path_to_annotation(path, LABELS, times, end), metric)
        matched, scored = frame_score.score_path(M, S, path)
        assert matched == pytest.approx(s.matched_s, abs=1e-9) and scored == pytest.approx(s.scored_s, abs=1e-9)
