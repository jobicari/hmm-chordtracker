"""The V0 report: figures and Markdown tables in one call each, for a notebook or a script.

    from chordtracker.report import v0_figures, results_tables
    figures = v0_figures(0.95, ["00_SS3-84-Bb_comp", "04_Funk2-119-G_comp"])   # shown inline in Jupyter
    print(results_tables("v0"))                                                # paste into reports/v0.md

Rule: test is scored, never inspected. Every figure here is drawn from the val split (results/v0/val/
and the val audio); the only test numbers are the pooled scores in results_tables, read from
results/v0/test/summary.json. A test take is rejected by v0_figures, so the analysis that shapes the
next version can never look at test. Nothing is chosen or rescored here: p_self must be the value
already chosen on val.
"""
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from chordtracker import emission, evaluate, experiment, paths, visualize
from chordtracker.data.data import load_manifest, load_split
from chordtracker.decode import FramewiseDecoder, ViterbiDecoder
from chordtracker.features import ChromaCQTExtractor

MANIFEST = paths.DATA_DIR / "processed" / "guitarset" / "manifest.csv"
SPLIT = paths.SPLITS_DIR / "guitarset.json"
CONFIG = paths.REPO_ROOT / "configs" / "v0.yaml"
ANALYSIS_SPLIT = "val"


def load_summary(version: str, split: str) -> dict:
    """results/<version>/<split>/summary.json as a dict."""
    return json.loads((paths.REPO_ROOT / "results" / version / split / "summary.json").read_text())


def load_per_song(version: str, split: str) -> list[dict]:
    """Rows of results/<version>/<split>/per_song.csv (values as text)."""
    with open(paths.REPO_ROOT / "results" / version / split / "per_song.csv") as file:
        return list(csv.DictReader(file))


def v0_figures(p_self: float, takes: list[str], config_path: Path = CONFIG) -> dict:
    """Build the V0 figures from val: {"sweep", "per_take", "groups", "durations", "take_<song_id>"...} -> Figure.

    p_self   the value chosen on val (0.95 for V0)
    takes    val song_ids for the one-take figures, e.g. ["00_SS3-84-Bb_comp", "04_Funk2-119-G_comp"]
    The take and duration figures recompute the val posteriors from the audio (about 80 s).
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    hop_seconds = cfg["features"]["hop_length"] / cfg["audio"]["sr"]
    song_ids = set(load_split(SPLIT)[ANALYSIS_SPLIT])
    outside = [take for take in takes if take not in song_ids]
    if outside:
        raise ValueError(f"{outside} not in the {ANALYSIS_SPLIT} split: figures never inspect test takes")

    extractor = ChromaCQTExtractor.from_config(cfg)
    model = emission.TemplateEmission.from_config(cfg)
    records = [record for record in load_manifest(MANIFEST) if record.song_id in song_ids]
    decoders = {"framewise": FramewiseDecoder(), p_self: ViterbiDecoder(p_self)}
    durations = {"reference": [], "framewise": [], p_self: []}
    figures = {}
    for record in records:
        evidence = experiment.take_evidence(record, extractor, model)
        durations["reference"].append(visualize.segment_durations(evidence.reference))
        decoded = {system: decoder.decode(evidence.posteriors) for system, decoder in decoders.items()}
        for system, path in decoded.items():
            estimate = evaluate.path_to_annotation(path, model.labels, evidence.frame_times, evidence.duration)
            durations[system].append(visualize.segment_durations(estimate))
        if record.song_id in takes:
            figures[f"take_{record.song_id}"] = visualize.plot_take(
                extractor.extract(record.audio), evidence.posteriors, model.labels, evidence.frame_times,
                evidence.reference, decoded, title=record.song_id)

    per_song_rows = load_per_song(cfg["version"], ANALYSIS_SPLIT)
    figures["sweep"] = visualize.plot_sweep(load_summary(cfg["version"], ANALYSIS_SPLIT)["rows"])
    figures["per_take"] = visualize.plot_per_take(per_song_rows, p_self)
    figures["groups"] = visualize.plot_groups(per_song_rows, p_self)
    figures["durations"] = visualize.plot_durations(
        {source: np.concatenate(values) for source, values in durations.items()}, hop_seconds)
    return figures


def save_figures(figures: dict, version: str) -> Path:
    """Save every figure to figures/<version>/val/<name>.png and return that folder."""
    figures_dir = paths.REPO_ROOT / "figures" / version / ANALYSIS_SPLIT
    for name, fig in figures.items():
        visualize.save(fig, figures_dir / f"{name}.png")
    return figures_dir


def results_tables(version: str) -> str:
    """Markdown tables read from the saved results, so the report never holds a hand-typed number.

    1. val sweep: pooled WCSR and changes/min per system, the chosen p_self in bold
    2. val by style: pooled WCSR per style, framewise vs the chosen p_self
    3. test, run once: pooled WCSR, changes/min, takes improved, reduced and excluded share
    """
    val = load_summary(version, "val")
    test = load_summary(version, "test")
    chosen = val["chosen_p_self"]

    lines = ["| System | val WCSR | changes / min |", "|---|---|---|"]
    for row in val["rows"]:
        name = visualize.system_name(row["system"])
        if row["system"] == chosen:
            name = f"**{name}** (chosen)"
        lines.append(f"| {name} | {100 * row['wcsr']:.1f} % | {row['mean_changes_per_min']:.0f} |")

    per_song = load_per_song(version, "val")
    framewise = style_pooled(per_song, "framewise")
    viterbi = style_pooled(per_song, chosen)
    lines += ["", f"| Style | framewise | Viterbi {chosen:g} |", "|---|---|---|"]
    lines += [f"| {style} | {100 * framewise[style]:.1f} % | {100 * viterbi[style]:.1f} % |" for style in framewise]

    test_rows = load_per_song(version, "test")
    helped = takes_improved(test_rows, chosen)
    lines += ["", "| System | test WCSR | changes / min | takes improved | reduced | excluded |",
              "|---|---|---|---|---|---|"]
    for row in test["rows"]:
        total = row["scored_s"] + row["excluded_s"]
        improved = "" if row["system"] == "framewise" else f"{helped[0]} / {helped[1]}"
        lines.append(f"| {visualize.system_name(row['system'])} | {100 * row['wcsr']:.1f} % | "
                     f"{row['mean_changes_per_min']:.0f} | {improved} | {100 * row['reduced_s'] / row['scored_s']:.1f} % "
                     f"| {100 * row['excluded_s'] / total:.1f} % |")
    return "\n".join(lines)


def style_pooled(per_song_rows: list[dict], system) -> dict[str, float]:
    """Pooled WCSR per style (the group without its progression number): {"BN": 0.562, "Funk": 0.291, ...}."""
    matched, scored = {}, {}
    for row in per_song_rows:
        if visualize.same_system(row["system"], system):
            style = visualize.group_of(row["song_id"]).rstrip("0123456789")
            matched[style] = matched.get(style, 0.0) + float(row["matched_s"])
            scored[style] = scored.get(style, 0.0) + float(row["scored_s"])
    return {style: matched[style] / scored[style] for style in sorted(scored)}


def takes_improved(per_song_rows: list[dict], system) -> tuple[int, int]:
    """(takes where the system beats framewise, number of takes)."""
    framewise = {row["song_id"]: float(row["wcsr"]) for row in per_song_rows if visualize.same_system(row["system"], "framewise")}
    other = {row["song_id"]: float(row["wcsr"]) for row in per_song_rows if visualize.same_system(row["system"], system)}
    return sum(other[song_id] > framewise[song_id] for song_id in framewise), len(framewise)
