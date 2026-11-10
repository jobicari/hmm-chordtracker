"""Component 41 (V15): decoding and scoring vocabulary A+ on validation, for training and for the tables.

    song_tables(take, n_frames, vocab, cfg, metrics)    {metric_M, metric_S}: frame_score tables over vocab's
                                                        decode labels (the chords, then N; reference N scored)
    decoder(vocab_qualities, counts, prior, mix_prior, p_self, chord_to_n, n_weight, power)
                                                        (A, decode prior) for the fixed-lag Viterbi
    decode(post, A, prior, lag)                         (T,) path in decode order; lag None = frame argmax
    from_v14(path, vocab)                               a V14 path (61 classes, N = 60) in vocab's decode order
    score(paths, songs, metrics, vocab, period)         per set (and their mean) WCSR % for every metric, plus
                                                        per-quality recall / precision over all val

Everything here is in decode order (data_v15.Vocab.decode_order): N is the last class, so V14's decoder
pieces (transitions_v8.change_matrix with chord_to_n, tempered_prior, the N weight) apply unchanged.

Per-quality recall and precision are frame counts, pooled over the three val sets (the new qualities
have too few seconds in each set alone), from the frame targets of vocabulary A+ (a frame is scored
by the chord at its centre):
    recall(q)     frames whose reference is quality q and whose output is the same class (root and quality)
                  / frames whose reference is quality q
    precision(q)  frames output as quality q whose reference is that class / frames output as quality q
                  that have a reference (N or a chord of A+; X and chords outside A+ are left out)
Seconds = frames * hop / sr (10.77 frames per second).
"""
import numpy as np

from chordtracker import frame_score, online_hmm, transitions, transitions_v8
from chordtracker.data_v15 import Vocab

SETS = ("rwc", "guitarset", "choco")
METRICS = ("majmin", "sevenths", "tetrads", "mirex")


def song_tables(annotation, n_frames: int, vocab: Vocab, period: float,
                metrics: tuple[str, ...] = METRICS) -> dict[str, np.ndarray]:
    """{f'{m}_M', f'{m}_S'} (n_frames, n_classes) float32 tables over vocab.decode_names(), reference N scored."""
    labels = vocab.decode_names()
    out = {}
    for m in metrics:
        M, S = frame_score.tables(annotation, labels, n_frames, period, m)
        out[f"{m}_M"], out[f"{m}_S"] = M.astype(np.float32), S.astype(np.float32)
    return out


def decoder(qualities: tuple[str, ...], counts: np.ndarray, prior: np.ndarray, divide_prior: np.ndarray,
            p_self: float, chord_to_n: float, n_weight: float, power: float, alpha: float = 1.0,
            tau: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(A, decode prior) as V14's run_v14.specs: the Billboard matrix split by the prior's quality shares,
    N's row and column at chord_to_n, p_self on the diagonal; the evidence divides by divide_prior ** power,
    N's entry divided by n_weight (more N evidence for n_weight > 1). A class divide_prior gives no mass is
    divided by the smallest mass it gives a class (not by ~0, which would make its evidence unbounded)."""
    Q = transitions_v8.change_matrix(counts, qualities, prior, alpha, tau, chord_to_n)
    floor = divide_prior[divide_prior > 0].min()
    d = transitions_v8.tempered_prior(np.maximum(divide_prior, floor), power)
    d[-1] /= n_weight
    return transitions.transition(Q, p_self), d / d.sum()


def decode(post: np.ndarray, A: np.ndarray | None, prior: np.ndarray | None, lag: int | None) -> np.ndarray:
    """(T,) decode-order path: the frame argmax for lag None, else fixed-lag Viterbi at that lag."""
    if lag is None:
        return post.argmax(axis=0)
    return online_hmm.fixed_lag_viterbi(post, A, prior, [lag])[lag]


def from_v14(path: np.ndarray, vocab: Vocab) -> np.ndarray:
    """A V14 path (0 .. 59 chords of A, 60 N) as decode indices of vocab (N -> vocab.n_chords)."""
    return np.where(path == 60, vocab.n_chords, path)


def quality_blocks(vocab: Vocab) -> dict[str, np.ndarray]:
    """{quality: its 12 decode indices}."""
    return {q: np.arange(12 * i, 12 * i + 12) for i, q in enumerate(vocab.qualities)}


def score(paths: dict[str, np.ndarray], songs: dict[str, dict], vocab: Vocab, period: float,
          metrics: tuple[str, ...] = METRICS) -> dict:
    """{'sets': {set: {metric: %}}, 'mean': {metric: %}, 'quality': {q: {ref_s, out_s, recall, precision}}}.

    songs[key]: 'set', 'targets' (decode order, -1 = none) and the tables of song_tables.
    """
    acc = {d: {m: [0.0, 0.0] for m in metrics} for d in SETS}
    blocks = quality_blocks(vocab)
    q_acc = {q: [0, 0, 0, 0] for q in blocks}                  # ref frames, ref right, out frames, out right
    for key, path in paths.items():
        s = songs[key]
        for m in metrics:
            matched, scored = frame_score.score_path(s[f"{m}_M"], s[f"{m}_S"], path)
            acc[s["set"]][m][0] += matched
            acc[s["set"]][m][1] += scored
        y = s["targets"]
        right = path == y
        for q, idx in blocks.items():
            ref_q, out_q = np.isin(y, idx), np.isin(path, idx) & (y >= 0)
            q_acc[q][0] += int(ref_q.sum())
            q_acc[q][1] += int((ref_q & right).sum())
            q_acc[q][2] += int(out_q.sum())
            q_acc[q][3] += int((out_q & right).sum())
    sets = {d: {m: 100 * v[0] / v[1] if v[1] else float("nan") for m, v in a.items()} for d, a in acc.items()}
    mean = {m: float(np.mean([sets[d][m] for d in SETS])) for m in metrics}
    quality = {q: {"ref_s": a[0] * period, "out_s": a[2] * period,
                   "recall": 100 * a[1] / a[0] if a[0] else float("nan"),
                   "precision": 100 * a[3] / a[2] if a[2] else float("nan")} for q, a in q_acc.items()}
    return {"sets": sets, "mean": mean, "quality": quality}


def new_recall(result: dict, vocab: Vocab) -> float:
    """Mean recall % of the new qualities (each quality counts once, whatever its seconds)."""
    return float(np.mean([result["quality"][q]["recall"] for q in vocab.new]))
