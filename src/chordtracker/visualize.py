"""Component 7: figures for the V0 report, as plain functions.

Every function takes data that already exists (arrays, annotations, result rows) and returns a
matplotlib Figure; nothing here decodes or scores. Save with save(fig, path). No backend is forced,
so the same functions show inline in a Jupyter notebook and save files from a script.

    plot_take(chroma, posteriors, labels, frame_times, reference, paths)   one take, 3 stacked panels
    plot_sweep(summary_rows)                                              val WCSR and changes/min vs p_self
    plot_per_take(per_song_rows, system)                                  framewise vs Viterbi per take, one panel per group
    plot_groups(per_song_rows, system)                                    pooled WCSR per group, framewise vs Viterbi
    plot_durations(durations, hop_seconds)                                chord-duration histograms, reference vs estimates

Colours: one fixed colour per system, the same in every figure (framewise blue, Viterbi orange,
reference aqua), from the project's validated categorical palette. Magnitudes (chroma,
posteriors) use a single-hue blue ramp, light = 0, dark = 1. Two quantities with different units
never share an axis: they get two panels.
"""
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from chordtracker import evaluate
from chordtracker.data.data import ChordAnnotation

FRAMEWISE_COLOR = "#2a78d6"     # categorical slot 1, blue
VITERBI_COLOR = "#eb6834"       # categorical slot 2, orange
REFERENCE_COLOR = "#1baf7a"     # categorical slot 3, aqua (below 3:1 on white: always labelled in a legend)
GRID_COLOR = "#e6e5e1"
TEXT_COLOR = "#52514e"
MAGNITUDE_CMAP = LinearSegmentedColormap.from_list(
    "magnitude", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ====================================================================================
# ================================      HELPERS     ==================================
# ====================================================================================

def system_color(system) -> str:
    """"framewise" -> blue, any p_self (0.95) -> orange, "reference" -> aqua."""
    if system == "reference":
        return REFERENCE_COLOR
    return FRAMEWISE_COLOR if system == "framewise" else VITERBI_COLOR


def system_name(system) -> str:
    """"framewise" -> "framewise", 0.95 -> "Viterbi 0.95", "reference" -> "reference"."""
    return system if isinstance(system, str) else f"Viterbi {system:g}"


def group_of(song_id: str) -> str:
    """Style + progression of a GuitarSet take: "04_Funk1-114-Ab_comp" -> "Funk1"."""
    return song_id.split("_")[1].split("-")[0]


def segment_durations(annotation: ChordAnnotation) -> np.ndarray:
    """Length of every segment in seconds: [[0, 2], [2, 2.5]] -> [2.0, 0.5]."""
    return annotation.intervals[:, 1] - annotation.intervals[:, 0]


def same_system(row_system, system) -> bool:
    """Compare a "system" value from a CSV row ("0.95" as text) with "framewise" or 0.95."""
    if isinstance(system, str):
        return str(row_system) == system
    try:
        return float(row_system) == float(system)
    except ValueError:
        return False


def style_axes(ax) -> None:
    """Recessive frame: no top/right spines, light grid behind the data, muted tick labels."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def plain_log_ticks(axis, ticks) -> None:
    """Log axis with readable ticks: [25, 50, 100, 200] instead of 2.5 x 10^1 ... ; no minor ticks."""
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_locator(NullLocator())


def save(fig, path: Path | str, dpi=200) -> None:
    """Write the figure (PNG or PDF from the suffix), creating the folder: save(fig, "figures/v0/sweep.png")."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ====================================================================================
# ================================     ONE TAKE     ==================================
# ====================================================================================

def plot_take(chroma: np.ndarray, posteriors: np.ndarray, labels: list[str], frame_times: np.ndarray,
              reference: ChordAnnotation, paths: dict, title: str = ""):
    """Three panels sharing the time axis, top to bottom:

        chroma       (12, n_frames)  pitch class energy, C at the bottom
        posteriors   (n_states, n_frames)  P(chord | frame) output by the emission model, every row labelled
        chords       one row per chord that appears anywhere; reference as a wide aqua band,
                     each decoded path as a thin bar on top (framewise blue, Viterbi orange)

    paths maps a system to its state path: {"framewise": (n_frames,), 0.95: (n_frames,)}.
    Only the chord rows that occur in the reference or a path are drawn, so a take with
    6 chords shows 6 rows, not 25.
    """
    duration = float(reference.intervals[-1, 1])
    estimates = {system: evaluate.path_to_annotation(path, labels, frame_times, duration)
                 for system, path in paths.items()}
    used = sorted({*reference.labels, *(label for ann in estimates.values() for label in ann.labels)},
                  key=lambda label: labels.index(label) if label in labels else len(labels))
    row_of = {label: row for row, label in enumerate(used)}

    fig, (ax_chroma, ax_post, ax_chords) = plt.subplots(
        3, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [2, 5, max(2, 0.35 * len(used))]})
    extent = [0, duration, -0.5, None]

    ax_chroma.imshow(chroma, aspect="auto", origin="lower", cmap=MAGNITUDE_CMAP, vmin=0, vmax=1,
                     extent=[0, duration, -0.5, 11.5], interpolation="nearest")
    ax_chroma.set_yticks(range(12), PITCH_CLASSES)
    ax_chroma.set_ylabel("chroma", color=TEXT_COLOR)

    extent[3] = posteriors.shape[0] - 0.5
    ax_post.imshow(posteriors, aspect="auto", origin="lower", cmap=MAGNITUDE_CMAP, vmin=0,
                   vmax=float(posteriors.max()), extent=extent, interpolation="nearest")
    ax_post.set_yticks(range(len(labels)), labels, fontsize=6)
    ax_post.set_ylabel("P(chord | frame)", color=TEXT_COLOR)

    for (start, end), label in reference:
        if label in row_of:
            ax_chords.barh(row_of[label], end - start, left=start, height=0.8, color=REFERENCE_COLOR,
                           alpha=0.45, linewidth=0)
    for offset, (system, estimate) in zip((-0.15, 0.15), estimates.items()):
        for (start, end), label in estimate:
            ax_chords.barh(row_of[label] + offset, end - start, left=start, height=0.22,
                           color=system_color(system), linewidth=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=REFERENCE_COLOR, alpha=0.45)]
    handles += [plt.Rectangle((0, 0), 1, 1, color=system_color(system)) for system in estimates]
    ax_chords.legend(handles, ["reference", *map(system_name, estimates)], loc="upper center",
                     bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    ax_chords.set_yticks(range(len(used)), used)
    ax_chords.set_ylim(-0.6, len(used) - 0.4)
    ax_chords.set_xlabel("time (s)", color=TEXT_COLOR)
    ax_chords.set_xlim(0, duration)

    for ax in (ax_chroma, ax_post, ax_chords):
        ax.tick_params(colors=TEXT_COLOR, labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    ax_chords.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax_chords.set_axisbelow(True)
    if title:
        fig.suptitle(title, color=TEXT_COLOR, fontsize=10)
    fig.tight_layout()
    return fig


# ====================================================================================
# ================================     RESULTS      ==================================
# ====================================================================================

def plot_sweep(summary_rows: list[dict]):
    """Two panels against p_self (one per unit, never a second y-axis): pooled WCSR and changes/min.

    The Viterbi rows are the line; the framewise row is a dashed horizontal reference in both
    panels. p_self values sit at equal spacing (0.04, 0.5, 0.9, ..., 0.999) because the
    interesting ones crowd near 1.
    """
    viterbi = [row for row in summary_rows if row["system"] != "framewise"]
    framewise = next(row for row in summary_rows if row["system"] == "framewise")
    positions = np.arange(len(viterbi))

    fig, (ax_wcsr, ax_changes) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key, scale, ylabel in ((ax_wcsr, "wcsr", 100, "pooled WCSR (%)"),
                                   (ax_changes, "mean_changes_per_min", 1, "chord changes per minute")):
        values = [scale * row[key] for row in viterbi]
        ax.axhline(scale * framewise[key], color=FRAMEWISE_COLOR, linewidth=2, linestyle="--", label="framewise")
        ax.plot(positions, values, color=VITERBI_COLOR, linewidth=2, marker="o", markersize=5,
                markeredgecolor="white", markeredgewidth=1.5, label="Viterbi")
        ax.set_xticks(positions, [f"{row['system']:g}" for row in viterbi])
        ax.set_xlabel("p_self", color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        style_axes(ax)
    ax_changes.set_yscale("log")
    plain_log_ticks(ax_changes.yaxis, [10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 500])
    ax_wcsr.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    fig.tight_layout()
    return fig


def plot_per_take(per_song_rows: list[dict], system):
    """Small multiples, one panel per group: each dot is a take, framewise WCSR (x) vs Viterbi WCSR (y).

    Dots above the diagonal are takes where Viterbi helps, below where it hurts. One panel per
    group instead of one colour per group keeps every panel to a single colour.
    """
    framewise = {row["song_id"]: float(row["wcsr"]) for row in per_song_rows if same_system(row["system"], "framewise")}
    viterbi = {row["song_id"]: float(row["wcsr"]) for row in per_song_rows if same_system(row["system"], system)}
    groups = sorted({group_of(song_id) for song_id in framewise})

    fig, axes = plt.subplots(1, len(groups), figsize=(2.4 * len(groups), 2.8), sharex=True, sharey=True)
    for ax, group in zip(np.atleast_1d(axes), groups):
        song_ids = [song_id for song_id in framewise if group_of(song_id) == group]
        ax.plot([0, 100], [0, 100], color=GRID_COLOR, linewidth=1.5, zorder=1)
        ax.scatter([100 * framewise[s] for s in song_ids], [100 * viterbi[s] for s in song_ids], s=36,
                   color=VITERBI_COLOR, edgecolor="white", linewidth=1.5, zorder=2)
        helped = sum(viterbi[s] > framewise[s] for s in song_ids)
        ax.set_title(f"{group}  ({helped}/{len(song_ids)} helped)", color=TEXT_COLOR, fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal")
        style_axes(ax)
    first = np.atleast_1d(axes)[0]
    first.set_ylabel(f"{system_name(system)} WCSR (%)", color=TEXT_COLOR)
    fig.supxlabel("framewise WCSR (%)", color=TEXT_COLOR, fontsize=10)
    fig.tight_layout()
    return fig


def pooled_by_group(per_song_rows: list[dict], system) -> dict[str, float]:
    """Pooled WCSR per group for one system: {"BN2": 0.468, "Funk1": 0.305, ...} (sum matched / sum scored)."""
    matched, scored = defaultdict(float), defaultdict(float)
    for row in per_song_rows:
        if same_system(row["system"], system):
            matched[group_of(row["song_id"])] += float(row["matched_s"])
            scored[group_of(row["song_id"])] += float(row["scored_s"])
    return {group: matched[group] / scored[group] for group in sorted(scored) if scored[group] > 0}


def plot_groups(per_song_rows: list[dict], system):
    """Grouped bars: pooled WCSR per group, framewise (blue) next to Viterbi (orange), value on each bar."""
    framewise = pooled_by_group(per_song_rows, "framewise")
    viterbi = pooled_by_group(per_song_rows, system)
    groups = list(framewise)
    positions = np.arange(len(groups))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.4 * len(groups) + 2, 3.6))
    for offset, values, system_key in ((-width / 2, framewise, "framewise"), (width / 2, viterbi, system)):
        heights = [100 * values[group] for group in groups]
        bars = ax.bar(positions + offset, heights, width=width - 0.03, color=system_color(system_key),
                      label=system_name(system_key))
        ax.bar_label(bars, fmt="%.0f", fontsize=7, color=TEXT_COLOR, padding=2)
    ax.set_xticks(positions, groups)
    ax.set_ylabel("pooled WCSR (%)", color=TEXT_COLOR)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    style_axes(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def frame_bins(hop_seconds: float, longest: float = 30.0, n_bins: int = 40) -> np.ndarray:
    """Log-spaced histogram edges snapped to whole frames, so no bin is narrower than one frame.

    Segment lengths come in whole frames (hop 512 at 22050 Hz = 23.2 ms), so plain log spacing
    leaves empty bins between 1, 2, 3 frames that draw a comb. Edges sit halfway between frame
    counts: with hop 0.1 s -> 0.05, 0.15, 0.25, ... for short lengths, then log-spaced but rounded.
    """
    frames = np.unique(np.round(np.geomspace(1, longest / hop_seconds, n_bins)))
    return np.concatenate([[frames[0] - 0.5], frames + 0.5]) * hop_seconds


def plot_durations(durations: dict, hop_seconds: float = 512 / 22050):
    """Histogram of chord-segment lengths, one step line per source, log-spaced bins snapped to whole frames.

    durations maps a source to all its segment lengths over the split:
        {"reference": (n_ref,), "framewise": (n_fw,), 0.95: (n_vit,)}
    The y-axis is the share of that source's segments (each line sums to 100%), because framewise has
    about ten times more segments than the others and would flatten them on raw counts. Framewise
    piles up at one frame (23 ms), since a single-frame blip is a segment; Viterbi should sit near the
    reference.
    """
    bins = frame_bins(hop_seconds)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for source, values in durations.items():
        ax.hist(np.clip(values, bins[0], bins[-1]), bins=bins, weights=np.full(len(values), 100 / len(values)),
                histtype="step", linewidth=2, color=system_color(source),
                label=f"{system_name(source)} ({len(values)} segments)")
    ax.set_xscale("log")
    plain_log_ticks(ax.xaxis, [0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25])
    ax.set_xlabel("segment length (s)", color=TEXT_COLOR)
    ax.set_ylabel("share of segments (%)", color=TEXT_COLOR)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_COLOR)
    style_axes(ax)
    fig.tight_layout()
    return fig
