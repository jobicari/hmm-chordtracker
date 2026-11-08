"""Behaviour of run_v12: the offsets step runs V10's method on V11's pool students and writes to results/v12."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_v10  # noqa: E402
import run_v12  # noqa: E402
import v10_offsets  # noqa: E402


def test_frozen_p_self_reads_the_v11_frozen_points():
    assert run_v12.frozen_p_self(2) == 0.3
    assert run_v12.frozen_p_self(4) == 0.7
    assert run_v12.frozen_p_self(8) == 0.98


def test_offsets_step_points_at_the_v11_pool_students(monkeypatch):
    seen = {}
    monkeypatch.setattr(run_v10, "config", lambda: {"cfg": True})
    monkeypatch.setattr(run_v10, "load_songs", lambda: {"song": {}})
    monkeypatch.setattr(v10_offsets, "run", lambda cfg, songs, **kw: seen.update(kw))
    run_v12.main(["offsets"])
    assert {s: p.name for s, p in seen["seeds"].items()} == {0: "pool_kd1_seed0", 1: "pool_kd1_seed1"}
    assert seen["seeds"][1].parent.name == "v11"
    assert seen["share_p_self"] == 0.98 and seen["ensembles"] is False and seen["name"] == "offsets"


def test_write_goes_to_results_v12(monkeypatch, tmp_path):
    monkeypatch.setattr(run_v10.paths, "REPO_ROOT", tmp_path)
    run_v12.write("offsets", "| table |", {"a": np.float64(1.0)})
    assert (tmp_path / "results" / "v12" / "offsets" / "table.md").read_text() == "| table |\n"
    assert (tmp_path / "results" / "v12" / "offsets" / "results.json").exists()


def test_share_table_names_the_set():
    rows = {"seed 0": {"reference": [50, 30, 10, 5, 5], "network": [60, 30, 5, 3, 2], "lag 8": [62, 30, 4, 2, 2]}}
    text = v10_offsets.share_table(rows)
    assert text.startswith("| RWC-P val, share of frames %")
    assert "| reference | 50.0 | 30.0 | 10.0 | 5.0 | 5.0 |" in text and "| seed 0, lag 8 | 62.0" in text


def test_tuning_match_names_the_estimate_the_cache_agrees_with():
    import v12_tuning
    assert v12_tuning.matched(5e-5, 0.05) == "24 bins"
    assert v12_tuning.matched(0.05, 5e-5) == "12 bins"
    assert v12_tuning.matched(5e-5, 6e-5) == "both"
    assert v12_tuning.matched(0.05, 0.06) == "neither"
