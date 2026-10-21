"""Tests for chordtracker.visualize (component 7).

Figures are not compared pixel by pixel: the tests check the helpers exactly and that every plot
function draws the expected panels and saves a non-empty PNG from small hand-made data.
"""
import numpy as np
import pytest

from chordtracker import visualize
from chordtracker.data.data import ChordAnnotation

LABELS = ["C:maj", "G:maj", "A:min", "N"]


def per_song_rows():
    """Two groups, two takes each, framewise and Viterbi 0.95 rows as run_v0.py writes them (text values)."""
    rows = []
    for song_id, framewise, viterbi in (("00_BN2-1-C_comp", 4, 6), ("01_BN2-1-C_comp", 5, 7),
                                        ("00_Funk1-1-C_comp", 6, 3), ("01_Funk1-1-C_comp", 2, 4)):
        for system, matched in (("framewise", framewise), ("0.95", viterbi)):
            rows.append({"system": system, "song_id": song_id, "wcsr": str(matched / 10),
                         "matched_s": str(matched), "scored_s": "10"})
    return rows


# ------------------------------------------------------------------------ helpers

def test_group_of():
    """"04_Funk1-114-Ab_comp" -> "Funk1", "00_SS2-88-F_comp" -> "SS2"."""
    assert visualize.group_of("04_Funk1-114-Ab_comp") == "Funk1"
    assert visualize.group_of("00_SS2-88-F_comp") == "SS2"


def test_segment_durations():
    """Segments [0, 2], [2, 2.5] -> lengths [2.0, 0.5]."""
    annotation = ChordAnnotation(np.array([[0.0, 2.0], [2.0, 2.5]]), ["C:maj", "G:maj"])
    np.testing.assert_allclose(visualize.segment_durations(annotation), [2.0, 0.5])


def test_system_colors_and_names_are_fixed():
    """framewise is always blue, any p_self orange, reference aqua; 0.95 is named "Viterbi 0.95"."""
    assert visualize.system_color("framewise") == visualize.FRAMEWISE_COLOR
    assert visualize.system_color(0.95) == visualize.system_color(0.5) == visualize.VITERBI_COLOR
    assert visualize.system_color("reference") == visualize.REFERENCE_COLOR
    assert visualize.system_name(0.95) == "Viterbi 0.95" and visualize.system_name("framewise") == "framewise"


def test_same_system_reads_csv_text():
    """A CSV row says "0.95": it matches 0.95 but not 0.9 or "framewise"."""
    assert visualize.same_system("0.95", 0.95)
    assert not visualize.same_system("0.95", 0.9) and not visualize.same_system("0.95", "framewise")
    assert visualize.same_system("framewise", "framewise")


def test_frame_bins_are_never_narrower_than_a_frame():
    """With hop 0.1 s the first edges are 0.05, 0.15, 0.25 (one frame wide each) and every bin holds a whole frame count."""
    bins = visualize.frame_bins(0.1, longest=30.0)
    np.testing.assert_allclose(bins[:3], [0.05, 0.15, 0.25])
    assert (np.diff(bins) >= 0.1 - 1e-9).all()
    frame_counts = bins / 0.1 - 0.5
    np.testing.assert_allclose(frame_counts, np.round(frame_counts), atol=1e-9)


def test_pooled_by_group():
    """BN2 framewise (4 + 5) / 20 = 0.45, Viterbi (6 + 7) / 20 = 0.65; Funk1 framewise 0.4, Viterbi 0.35."""
    rows = per_song_rows()
    assert visualize.pooled_by_group(rows, "framewise") == pytest.approx({"BN2": 0.45, "Funk1": 0.4})
    assert visualize.pooled_by_group(rows, 0.95) == pytest.approx({"BN2": 0.65, "Funk1": 0.35})


# ------------------------------------------------------------------------ figures

def test_plot_take(tmp_path):
    """3 stacked panels; every state is labelled in the posteriors panel, the chord panel only shows the 3 that occur."""
    rng = np.random.default_rng(0)
    frame_times = np.arange(20) * 0.1
    reference = ChordAnnotation(np.array([[0.0, 1.0], [1.0, 2.0]]), ["C:maj", "G:maj"])
    paths = {"framewise": np.array([0] * 9 + [2] + [1] * 10), 0.95: np.array([0] * 10 + [1] * 10)}
    fig = visualize.plot_take(rng.random((12, 20)), rng.random((4, 20)), LABELS, frame_times, reference, paths)
    assert len(fig.axes) == 3
    assert [tick.get_text() for tick in fig.axes[1].get_yticklabels()] == LABELS
    assert [tick.get_text() for tick in fig.axes[2].get_yticklabels()] == ["C:maj", "G:maj", "A:min"]
    visualize.save(fig, tmp_path / "take.png")
    assert (tmp_path / "take.png").stat().st_size > 0


def test_plot_sweep_uses_two_panels(tmp_path):
    """WCSR and changes/min get one panel each (no second y-axis), with one tick per p_self."""
    rows = [{"system": "framewise", "wcsr": 0.48, "mean_changes_per_min": 282.0},
            {"system": 0.5, "wcsr": 0.60, "mean_changes_per_min": 41.0},
            {"system": 0.95, "wcsr": 0.61, "mean_changes_per_min": 31.0}]
    fig = visualize.plot_sweep(rows)
    assert len(fig.axes) == 2
    assert [tick.get_text() for tick in fig.axes[0].get_xticklabels()] == ["0.5", "0.95"]
    visualize.save(fig, tmp_path / "sweep.png")
    assert (tmp_path / "sweep.png").stat().st_size > 0


def test_plot_per_take_has_one_panel_per_group(tmp_path):
    """Groups BN2 and Funk1 -> 2 panels; BN2 titles "2/2 helped", Funk1 "1/2 helped"."""
    fig = visualize.plot_per_take(per_song_rows(), 0.95)
    assert [ax.get_title() for ax in fig.axes] == ["BN2  (2/2 helped)", "Funk1  (1/2 helped)"]
    visualize.save(fig, tmp_path / "per_take.png")
    assert (tmp_path / "per_take.png").stat().st_size > 0


def test_plot_groups_and_durations_save(tmp_path):
    """The group bars and the duration histogram both draw and save a non-empty PNG."""
    visualize.save(visualize.plot_groups(per_song_rows(), 0.95), tmp_path / "groups.png")
    durations = {"reference": np.array([2.0, 2.0, 4.0]), "framewise": np.array([0.023, 0.05, 1.0]),
                 0.95: np.array([1.5, 2.5])}
    visualize.save(visualize.plot_durations(durations), tmp_path / "durations.png")
    assert (tmp_path / "groups.png").stat().st_size > 0 and (tmp_path / "durations.png").stat().st_size > 0


# ------------------------------------------------------------------------ report

def test_importing_visualize_does_not_force_a_backend():
    """visualize must not call matplotlib.use(): a notebook keeps its inline backend after the import."""
    import inspect
    assert "matplotlib.use(" not in inspect.getsource(visualize)


def test_save_figures_writes_everything_to_val(tmp_path, monkeypatch):
    """Every figure goes to figures/v0/val/<name>.png: the analysis never writes a test figure."""
    import matplotlib.pyplot as plt
    from chordtracker import paths, report

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    figures = {name: plt.figure() for name in ("sweep", "take_00_X-1-C_comp", "groups")}
    figures_dir = report.save_figures(figures, "v0")
    assert figures_dir == tmp_path / "figures" / "v0" / "val"
    assert sorted(path.name for path in figures_dir.iterdir()) == ["groups.png", "sweep.png", "take_00_X-1-C_comp.png"]


def test_v0_figures_refuses_a_test_take(tmp_path):
    """A take outside val is rejected before any audio is read: figures never inspect test."""
    import json
    from chordtracker import report

    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": [], "val": ["00_A-1-C_comp"], "test": ["00_B-1-C_comp"]}))
    original, report.SPLIT = report.SPLIT, split
    try:
        with pytest.raises(ValueError, match="never inspect test"):
            report.v0_figures(0.95, ["00_B-1-C_comp"])
    finally:
        report.SPLIT = original


def write_results(root, split, summary_rows, per_song):
    """results/v0/<split>/summary.json and per_song.csv under root, as run_v0.py writes them."""
    import csv
    import json
    out = root / "results" / "v0" / split
    out.mkdir(parents=True)
    summary = {"rows": summary_rows}
    if split == "val":
        summary["chosen_p_self"] = 0.95
    (out / "summary.json").write_text(json.dumps(summary))
    with open(out / "per_song.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["system", "song_id", "wcsr", "matched_s", "scored_s"])
        writer.writeheader()
        writer.writerows(per_song)


def test_results_tables_reads_the_saved_numbers(tmp_path, monkeypatch):
    """Val sweep with the chosen p_self in bold, val WCSR per style, test with takes improved and reduced/excluded shares."""
    from chordtracker import paths, report

    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path)
    row = {"scored_s": 90.0, "reduced_s": 9.0, "excluded_s": 10.0}
    rows = [{"system": "framewise", "wcsr": 0.4, "mean_changes_per_min": 280.0, **row},
            {"system": 0.95, "wcsr": 0.6, "mean_changes_per_min": 30.0, **row}]
    write_results(tmp_path, "val", rows, per_song_rows())
    write_results(tmp_path, "test", rows, per_song_rows())
    tables = report.results_tables("v0")
    assert "| **Viterbi 0.95** (chosen) | 60.0 % | 30 |" in tables
    assert "| BN | 45.0 % | 65.0 % |" in tables
    assert "| Funk | 40.0 % | 35.0 % |" in tables
    assert "| Viterbi 0.95 | 60.0 % | 30 | 3 / 4 | 10.0 % | 10.0 % |" in tables
