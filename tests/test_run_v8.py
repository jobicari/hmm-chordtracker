"""Behaviour of run_v8's scoring helpers: labels as scored, quality confusion, pooled summaries."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v8  # noqa: E402


def test_scored_labels_read_maj6_as_maj():
    labels = run_v8.class_labels("B")
    assert len(labels) == 84 and labels[60] == "C:maj" and labels[72] == "C:sus4"
    assert len(run_v8.class_labels("majmin")) == 24
    assert len(run_v8.class_labels("B9")) == 96 and run_v8.class_labels("B9")[84] == "C:9"


def test_confusion_counts_qualities_and_skips_frames_without_target():
    targets = np.array([0, 24, 24, -1, 37])
    path = np.array([0, 0, 25, 3, 48])
    c = run_v8.confusion(targets, path, 5)
    assert c[0, 0] == 1 and c[2, 0] == 1 and c[2, 2] == 1 and c[3, 4] == 1 and c.sum() == 4


def test_summarize_pools_per_set_and_averages_the_sets():
    def row(dataset, scored, matched):
        return {"dataset": dataset, "group": dataset, "system": "frame-argmax", "p_self": None, "lag": 0,
                "song_id": "x", "changes_per_min": 10.0, "majmin_scored_s": scored, "majmin_matched_s": matched,
                "sevenths_scored_s": scored, "sevenths_matched_s": matched / 2}
    rows = run_v8.summarize([row("rwc", 10, 5), row("rwc", 30, 25), row("choco", 10, 10)])
    by = {r["dataset"]: r for r in rows}
    assert by["rwc"]["majmin"] == 0.75 and by["choco"]["majmin"] == 1.0
    assert by["mean"]["majmin"] == 0.875 and by["mean"]["sevenths"] == 0.4375


def test_train_source_builds_q_from_the_label_counts():
    counts = np.zeros((5, 5, 12))
    counts[0, 2, 0] = 9
    cfg = {"transitions": {"source": "train", "alpha": 1, "tau": 1}}
    Q = run_v8.decoder_parts(cfg, "A", np.full(60, 1 / 60), counts)
    assert Q.shape == (60, 60) and Q[0, 24] == Q[0].max()
    assert run_v8.results_dir(cfg) == "val_train_tau1"
    assert run_v8.results_dir({"transitions": {"tau": 1}}) == "val"


def test_prior_power_names_its_results_folder():
    assert run_v8.results_dir({"transitions": {"tau": 1}, "decoder": {"prior_power": 0.5}}) == "val_pp0.5"
    assert run_v8.results_dir({"transitions": {"tau": 1}, "decoder": {"prior_power": 1}}) == "val"
