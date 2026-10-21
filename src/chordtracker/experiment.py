"""Component 6: running an experiment over a split, as plain functions.

The run script (scripts/run_v0.py) is a thin command line around these functions, so each step
can be tested on its own:

    take_evidence(record, extractor, emission)                 one take -> TakeEvidence (expensive, once)
    score_path(path, labels, evidence)                         one decoded path -> scores for that take
    sweep_p_self(takes, labels, grid)                          every p_self on cached evidence -> rows
    choose_p_self(rows)                                        the p_self with the best pooled WCSR
    write_results(summary_rows, per_song_rows, out_dir)        summary.json + per_song.csv

Protocol (PROJECT.md section 6): sweep p_self on the val split, choose it there, then run the test
split once with framewise and the chosen p_self. Features and posteriors do not depend on p_self, so
they are computed once per take and every decoder reuses them: a 7-value sweep costs one feature pass.
"""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chordtracker import decode, evaluate
from chordtracker.data.data import ChordAnnotation, TakeRecord

FRAMEWISE = "framewise"


# ====================================================================================
# ================================     EVIDENCE     ==================================
# ====================================================================================

@dataclass(frozen=True)
class TakeEvidence:
    """Everything a decoder and the scoring need for one take, computed once.

    posteriors   (n_states, n_frames), e.g. (25, 962) for a 22.3 s take
    frame_times  (n_frames,) seconds, frame t centred at t * hop / sr
    reference    the lead-sheet ChordAnnotation read from the take's .lab file
    duration     seconds = end of the last reference interval (22.3244 s for 00_BN1-129-Eb_comp)
    """
    song_id: str
    posteriors: np.ndarray
    frame_times: np.ndarray
    reference: ChordAnnotation
    duration: float


def take_evidence(record: TakeRecord, extractor, emission) -> TakeEvidence:
    """Load one take and compute its posteriors: chroma -> emission model -> (n_states, n_frames).

    record.audio and record.lab are full paths: load_manifest already joins them to the manifest's folder.
    """
    chroma = extractor.extract(record.audio)
    reference = ChordAnnotation.from_lab(record.lab)
    return TakeEvidence(song_id=record.song_id,
                        posteriors=emission.posteriors(chroma),
                        frame_times=extractor.frame_times(chroma.shape[1]),
                        reference=reference,
                        duration=float(reference.intervals[-1, 1]))


# ====================================================================================
# ================================      SCORING     ==================================
# ====================================================================================

def score_path(path: np.ndarray, labels: list[str], evidence: TakeEvidence) -> tuple[evaluate.SongScore, float]:
    """Score one decoded path against its take's reference: (SongScore, changes per minute).

    Example: a path that holds A#:maj for the whole of a 22.3 s take whose reference is A#:maj for
    the first 11 s -> wcsr about 0.49, 0 changes per minute.
    """
    estimate = evaluate.path_to_annotation(path, labels, evidence.frame_times, evidence.duration)
    return evaluate.score(evidence.reference, estimate), evaluate.changes_per_minute(estimate, evidence.duration)


def decoders_for(grid: list[float]) -> dict:
    """{"framewise": FramewiseDecoder(), 0.5: ViterbiDecoder(0.5), 0.9: ViterbiDecoder(0.9), ...}."""
    return {FRAMEWISE: decode.FramewiseDecoder(), **{p_self: decode.ViterbiDecoder(p_self) for p_self in grid}}


def sweep_p_self(takes: list[TakeEvidence], labels: list[str], grid: list[float]) -> tuple[list[dict], list[dict]]:
    """Decode every take with framewise and with Viterbi at every p_self of the grid.

    Returns (summary_rows, per_song_rows):
        summary_rows   one row per system: {"system": "framewise" or p_self, "wcsr": pooled WCSR,
                       "mean_changes_per_min", "scored_s", "reduced_s", "excluded_s", "n_takes"}
        per_song_rows  one row per (system, take): {"system", "song_id", "wcsr", "changes_per_min",
                       "total_s", "scored_s", "matched_s", "reduced_s", "excluded_s"}
    With a grid of 7 values and 60 takes: 8 summary rows and 480 per-song rows.
    """
    summary_rows, per_song_rows = [], []
    for system, decoder in decoders_for(grid).items():
        scores, changes = [], []
        for evidence in takes:
            song_score, changes_per_minute = score_path(decoder.decode(evidence.posteriors), labels, evidence)
            scores.append(song_score)
            changes.append(changes_per_minute)
            per_song_rows.append({"system": system, "song_id": evidence.song_id, "wcsr": song_score.wcsr,
                                  "changes_per_min": changes_per_minute, "total_s": song_score.total_s,
                                  "scored_s": song_score.scored_s, "matched_s": song_score.matched_s,
                                  "reduced_s": song_score.reduced_s, "excluded_s": song_score.excluded_s})
        total = evaluate.pooled(scores)
        summary_rows.append({"system": system, "wcsr": total.wcsr, "mean_changes_per_min": float(np.mean(changes)),
                             "scored_s": total.scored_s, "reduced_s": total.reduced_s,
                             "excluded_s": total.excluded_s, "n_takes": len(takes)})
    return summary_rows, per_song_rows


def choose_p_self(summary_rows: list[dict]) -> float:
    """The p_self with the highest pooled WCSR among the Viterbi rows; a tie goes to the smaller p_self.

    Example: 0.9 -> 0.61, 0.99 -> 0.63, 0.999 -> 0.63  ->  0.99 (less smoothing for the same score).
    The framewise row is ignored: it is the baseline, not a candidate.
    """
    viterbi_rows = [row for row in summary_rows if row["system"] != FRAMEWISE]
    if not viterbi_rows:
        raise ValueError("no Viterbi rows to choose from")
    best = max(viterbi_rows, key=lambda row: (row["wcsr"], -row["system"]))
    return best["system"]


# ====================================================================================
# ================================      OUTPUT      ==================================
# ====================================================================================

def write_results(summary: dict, per_song_rows: list[dict], out_dir: Path) -> None:
    """Write out_dir/summary.json (config, split, rows, chosen p_self) and out_dir/per_song.csv.

    Example: results/v0/val/summary.json and results/v0/val/per_song.csv (480 rows + header).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "per_song.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(per_song_rows[0]))
        writer.writeheader()
        writer.writerows(per_song_rows)
