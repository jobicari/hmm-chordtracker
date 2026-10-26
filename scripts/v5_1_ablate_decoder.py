"""V5.1 decoder ablation on val: how much N the decoder lets through. No retraining, saved model only.

    PYTHONPATH=src python scripts/v5_1_ablate_decoder.py      # -> results/v5.1/ablations/decoder.json

The val run found 43 of 107.5 s of RWC-P N but made 109 s of false N (and 12 s on GuitarSet, which
has none). Two decoder settings push towards N, and both are tried here on the same beat evidence:

    q_in      Q[chord, N], the share of a chord's changes that go to N. V3's rule: 1/24 = 4.2%.
              RWC-P train: 95 of 8,076 changes = 1.2%. Grid: 4.2% (as run), 1.2%, 0.5%.
    n_prior   the prior the decoder divides N by, as a multiple of N's training share (1.75%).
              1 = as run; 2 and 4 lift N less (4 x 1.75% = 7%, about one major chord's 5.5%).
              The 24 chord entries are rescaled so the prior still sums to 1.

Every combination is decoded at every p_self of the config grid; the beat evidence is computed once.
Two reference rows come first: n_prior "none" (no prior division, uniform P(state)) at q_in 1/24, and
the network alone (argmax per beat, no HMM), which shows what the HMM adds on top.
Val only: run_v5.evaluation_sets("val").
"""
import itertools
import json
import sys

import numpy as np

from chordtracker import acoustic_hmm_v5_1, beats, data_v5_1, evaluate, paths, train_v5_1

sys.path.insert(0, str(paths.REPO_ROOT / "scripts"))
import run_v5      # noqa: E402
import run_v5_1    # noqa: E402

Q_IN = [1 / 24, 95 / 8076, 0.005]
N_PRIOR = [1.0, 2.0, 4.0]


def with_q_in(Q: np.ndarray, q_in: float) -> np.ndarray:
    """Q with Q[chord, N] = q_in; the 23 chord moves of each chord row keep their ratios and sum to 1 - q_in."""
    Q = Q.copy()
    chords = slice(0, data_v5_1.N_CLASS)
    Q[chords, chords] *= (1 - q_in) / Q[chords, chords].sum(axis=1, keepdims=True)
    Q[chords, data_v5_1.N_CLASS] = q_in
    return Q


def with_n_prior(prior: np.ndarray, factor: float) -> np.ndarray:
    """The prior with N's entry times factor and the 24 chord entries rescaled, so it still sums to 1."""
    p = prior.copy()
    p[data_v5_1.N_CLASS] = min(factor * prior[data_v5_1.N_CLASS], 0.5)
    p[:data_v5_1.N_CLASS] *= (1 - p[data_v5_1.N_CLASS]) / prior[:data_v5_1.N_CLASS].sum()
    return p


def main() -> None:
    cfg = run_v5.load_config(paths.REPO_ROOT / "configs" / "v5_1.yaml")
    out = paths.REPO_ROOT / "results" / cfg["version"]
    model = train_v5_1.load(out / "model.pt", cfg["train"]["dropout"])
    prior = np.array(json.loads((out / "prior.json").read_text()))
    Q = run_v5_1.change_prior(cfg)
    grid = cfg["decoder"]["p_self_grid"]
    evidences = {name: [run_v5_1.evidence(model, t, cfg) for t in takes]
                 for name, takes in run_v5.evaluation_sets("val").items()}

    for name, evs in evidences.items():
        scores = [evaluate.score(ev.reference, beats.segments_to_annotation(
            ev.posteriors.argmax(axis=0), acoustic_hmm_v5_1.LABELS, ev.edges)) for ev in evs]
        print(f"network alone (argmax per beat, no HMM)  {name}: {100 * evaluate.pooled(scores).wcsr:6.2f}%")

    rows = []
    settings = [(1 / 24, "none")] + list(itertools.product(Q_IN, N_PRIOR))
    for q_in, factor in settings:
        Qa = with_q_in(Q, q_in)
        pa = np.full(len(prior), 1 / len(prior)) if factor == "none" else with_n_prior(prior, factor)
        by_dataset = {name: run_v5_1.sweep(name, evs, Qa, pa, grid)[0] for name, evs in evidences.items()}
        for i, p_self in enumerate(grid):
            row = {"q_in": q_in, "n_prior": factor, "p_self": p_self}
            for name, summary in by_dataset.items():
                r = summary[i]
                row.update({f"{name}_wcsr": r["wcsr"], f"{name}_n_found_s": r["n_found_s"],
                            f"{name}_n_ref_s": r["n_ref_n_s"], f"{name}_false_n_s": r["n_false_n_s"]})
            row["mean_wcsr"] = float(np.mean([row[f"{n}_wcsr"] for n in evidences]))
            rows.append(row)
            print(f"q_in {100 * q_in:4.1f}%  n_prior {factor if factor == 'none' else f'x{factor:.0f}'}  p_self {p_self:.2f}  "
                  f"rwc {100 * row['rwc_wcsr']:6.2f}% (N {row['rwc_n_found_s']:5.1f}/{row['rwc_n_ref_s']:5.1f} s, "
                  f"false {row['rwc_false_n_s']:5.1f} s)  gs {100 * row['guitarset_wcsr']:6.2f}% "
                  f"(false {row['guitarset_false_n_s']:4.1f} s)  mean {100 * row['mean_wcsr']:6.2f}%", flush=True)

    (out / "ablations").mkdir(parents=True, exist_ok=True)
    (out / "ablations" / "decoder.json").write_text(json.dumps(rows, indent=2))
    best = max(rows, key=lambda r: r["mean_wcsr"])
    print(f"best: q_in {best['q_in']:.4f}, n_prior {best['n_prior']}, p_self {best['p_self']}, mean {100 * best['mean_wcsr']:.2f}%")


if __name__ == "__main__":
    main()
