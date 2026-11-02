"""Headline WCSR plus the same score with reference N left out, from the saved .lab files. Reads only.

    python scripts/refs/rescore_n.py val     (or test)   -> prints a table, writes results/refs/n_excluded_<split>.json

Both parts come from evaluate.score, never from cutting N out of the reference (a reference with holes
makes mir_eval extend the chord before each hole). Scoring a copy of the reference with every chord
replaced by X judges the N seconds alone, and N-excluded = (matched - matched on N) / (scored - N).

V6 stores durations, not predictions. It has no N class and its segments cover the whole reference, so
it matches 0 s of reference N exactly, and its N-excluded score is matched / (scored - N).
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_refs as s

split = sys.argv[1]
sets = s.run_v5.evaluation_sets(split)


def only_n(reference):
    return s.ChordAnnotation(reference.intervals, [c if c == "N" else "X" for c in reference.labels])


with open(s.V6_RUN / split / "per_song.csv") as f:
    v6 = {r["song_id"]: r for r in csv.DictReader(f) if r["system"] == str(s.V6_P_SELF)}

rows = []
for dataset, takes in sets.items():
    all_n = [s.ChordAnnotation(t.annotation.intervals, ["N"] * len(t.annotation.labels)) for t in takes]
    reference_n = [s.evaluate.score(only_n(t.annotation), estimate) for t, estimate in zip(takes, all_n)]
    n_s = sum(x.scored_s for x in reference_n)
    assert all(abs(x.matched_s - x.scored_s) < 1e-6 for x in reference_n)
    scored = sum(float(v6[t.key]["scored_s"]) for t in takes)
    matched = sum(float(v6[t.key]["matched_s"]) for t in takes)
    rows.append({"tool": "v6", "dataset": dataset, "n_songs": len(takes), "wcsr": matched / scored,
                 "wcsr_n_excluded": matched / (scored - n_s), "reference_n_s": n_s, "n_right_s": 0.0,
                 "scored_s": scored})
    for tool, datasets in s.PAIRS.items():
        if dataset not in datasets:
            continue
        full, on_n = [], []
        for t in takes:
            intervals, raw = s.read_lab(s.lab_path(tool, split, dataset, t.key.split("/", 1)[1]))
            keep = intervals[:, 1] > intervals[:, 0]
            estimate = s.ChordAnnotation(intervals[keep], [s.to_majmin(c)[0] for c, k in zip(raw, keep) if k])
            full.append(s.evaluate.score(t.annotation, estimate))
            on_n.append(s.evaluate.score(only_n(t.annotation), estimate))
        f, n = s.evaluate.pooled(full), s.evaluate.pooled(on_n)
        assert abs(n.scored_s - n_s) < 1e-6 and abs(f.scored_s - scored) < 1e-6
        rows.append({"tool": tool, "dataset": dataset, "n_songs": len(takes), "wcsr": f.wcsr,
                     "wcsr_n_excluded": (f.matched_s - n.matched_s) / (f.scored_s - n.scored_s),
                     "reference_n_s": n.scored_s, "n_right_s": n.matched_s, "scored_s": f.scored_s})

(s.RESULTS / f"n_excluded_{split}.json").write_text(json.dumps(rows, indent=2))
print(f"{'tool':<9}{'dataset':<11}{'WCSR':>8}{'N-excl':>8}{'ref N s':>9}{'N right s':>11}{'scored s':>10}")
for r in rows:
    print(f"{r['tool']:<9}{r['dataset']:<11}{100 * r['wcsr']:>8.2f}{100 * r['wcsr_n_excluded']:>8.2f}"
          f"{r['reference_n_s']:>9.1f}{r['n_right_s']:>11.1f}{r['scored_s']:>10.1f}")
