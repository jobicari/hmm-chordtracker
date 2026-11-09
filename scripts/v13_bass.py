"""V13 on validation: does the bass pull the chord? Measured on RWC-P, with the bass notes from its aligned MIDI.

    python scripts/v13_bass.py measure       # -> results/v13/bass_measure/{table.md, results.json}
    python scripts/v13_bass.py heads         # -> results/v13/bass_heads/{table.md, results.json}

measure: V11's frozen student (seed 1) at the frozen lag 8 on RWC-P validation. A held chord is a run
of the annotation with the same vocabulary-A chord (the bass ignored, so C:maj and C:maj/3 are one
held chord); N, X and chords outside the vocabulary are not held chords. A decoded change inside a
held chord is a change of the decoded chord more than MARGIN_S from the held chord's ends (a change
near an end is a late or early real change). It coincides with the bass when the MIDI bass line
(midi.bass_line: a new lowest bass note) changes within WINDOW_S of it. Chance: the share of all
frames inside held chords (away from their ends) that are within WINDOW_S of a bass change, i.e. what
the coincidence would be if the decoded changes ignored the bass. Songs without a bass part in the
MIDI are left out.

heads: V10's model with root, quality and bass heads (data/v10_results/heads_seed0), decode only, on the
three validation sets. Its chord head is decoded as V8 does (the network alone; fixed-lag Viterbi at
lag 8, Billboard matrix, prior power 0.25, p_self chosen on the mean sevenths of the three sets), and
two uses of its bass head are added:
    slash      the chord path unchanged; a frame's label gets the bass head's note as its bass when that
               note is a chord tone other than the root (C:maj -> C:maj/3), from the bass posterior
               averaged over the frames t - 8 .. t + 8 (known when frame t is decided at lag 8)
    bass-hold  a chord change into frame t is made dearer when the bass head moves there: the change
               probability (1 - p_self) is multiplied by 1 - gamma * beta_t, beta_t = 1 - sum_k
               q_t(k) q_{t-1}(k) (q the bass posterior: the chance two draws differ), so a move of the
               bass alone is less able to switch the chord
Scores: majmin and sevenths on all three sets; majmin_inv and sevenths_inv where the reference has its
bass (RWC-P, GuitarSet, and of ChoCo the Beatles validation albums, whose chordlab is in data/beatles);
changes per minute; on RWC-P the decoded changes inside held chords near a MIDI bass change (measure).
"""
import argparse
import json

import numpy as np

import mir_eval
import torch

import run_v10 as r
from chordtracker import (data_choco, data_v5, data_v7_2, data_v8, evaluate, midi, model_v6, model_v10, online_hmm,
                          paths, transitions, transitions_v8)
from chordtracker.data.data import ChordAnnotation
from chordtracker.data_v5 import Song

RUN = paths.REPO_ROOT / "results" / "v11" / "pool_kd1_seed1"
HEADS = paths.DATA_DIR / "v10_results" / "heads_seed0"
BEATLES = paths.DATA_DIR / "beatles" / "chordlab" / "The Beatles"
BEATLES_VAL = ("04_-_Beatles_for_Sale", "09_-_Magical_Mystery_Tour")    # ChoCo's validation albums
GAMMAS = [0.5, 0.9, 0.99]
TONES = {"maj": {4: "3", 7: "5"}, "min": {3: "b3", 7: "5"}, "7": {4: "3", 7: "5", 10: "b7"},
         "maj7": {4: "3", 7: "5", 11: "7"}, "min7": {3: "b3", 7: "5", 10: "b7"}}
MIDI_DIR = paths.DATA_DIR / "rwc" / "rwc-annotations" / "01_annotations_preprocessed" / "MIDI_aligned" / "RWC-P"
OUT = paths.REPO_ROOT / "results" / "v13"
LAG, P_SELF = 8, 0.98
MARGIN_S = 0.3
WINDOW_S = 0.15


def held_chords(annotation, vocab: str = "A") -> list[tuple[float, float, int]]:
    """(start, end, class) of every run of annotation segments with the same vocabulary class, bass ignored."""
    runs = []
    for (s, e), label in zip(annotation.intervals, annotation.labels):
        c = data_v8.target(label, vocab)
        if runs and runs[-1][2] == c and abs(runs[-1][1] - s) < 1e-6:
            runs[-1] = (runs[-1][0], e, c)
        else:
            runs.append((s, e, c))
    return [x for x in runs if x[2] >= 0]


def changes_inside(path: np.ndarray, held: list[tuple[float, float, int]], hop_s: float,
                   margin: float = MARGIN_S) -> list[tuple[float, int, int]]:
    """(time, chord before, chord after) of the decoded changes inside a held chord, away from its ends."""
    t = np.nonzero(np.diff(path))[0] + 1
    out = []
    for f in t:
        time = f * hop_s
        if any(s + margin < time < e - margin for s, e, _ in held):
            out.append((time, int(path[f - 1]), int(path[f])))
    return out


def near_any(times: np.ndarray, events: np.ndarray, window: float = WINDOW_S) -> np.ndarray:
    """(n,) bool: each time within `window` s of one of the (sorted) event times."""
    if len(events) == 0:
        return np.zeros(len(times), dtype=bool)
    i = np.clip(np.searchsorted(events, times), 1, len(events) - 1)
    return np.minimum(np.abs(events[i] - times), np.abs(events[i - 1] - times)) <= window


def decode_lag8(post, cfg, raw_prior):
    Q = r.run_v8.decoder_parts(cfg, r.VOCAB, raw_prior)
    prior = transitions_v8.tempered_prior(raw_prior, cfg["decoder"]["prior_power"])
    return online_hmm.fixed_lag_viterbi(post, transitions.transition(Q, P_SELF), prior, [LAG])[LAG]


def step_measure() -> None:
    cfg = r.config()
    sr, hop = cfg["audio"]["sr"], cfg["cqt"]["hop_length"]
    hop_s = hop / sr
    posts, raw_prior = r.load_posteriors(RUN), r.prior_of(RUN)
    rows, skipped = [], []
    for take in data_v5.rwc_takes("val"):
        notes = midi.bass_notes(midi.read(MIDI_DIR / f"{take.key.split('/')[1]}.mid"))
        if not notes:
            skipped.append(take.key)
            continue
        bass = midi.bass_line(notes)
        bass_t = np.array([t for t, _ in bass])
        bass_pc = np.array([p % 12 for _, p in bass])
        held = held_chords(take.annotation)
        path = decode_lag8(posts[take.key], cfg, raw_prior)
        minutes = len(path) * hop_s / 60
        inside = changes_inside(path, held, hop_s)
        times = np.array([t for t, _, _ in inside])
        hit = near_any(times, bass_t)
        # does the new decoded chord sit on the new bass note (its root = the bass note's pitch class)?
        on_bass = 0
        for (time, _, after), h in zip(inside, hit):
            if h:
                k = int(np.argmin(np.abs(bass_t - time)))
                on_bass += int(after % 12 == bass_pc[k])
        frames = np.arange(len(path)) * hop_s
        in_held = np.array([any(s + MARGIN_S < x < e - MARGIN_S for s, e, _ in held) for x in frames])
        chance = float(near_any(frames[in_held], bass_t).mean()) if in_held.any() else 0.0
        held_bass_moves = int(sum(np.count_nonzero((bass_t > s + MARGIN_S) & (bass_t < e - MARGIN_S))
                                  for s, e, _ in held))
        # how often the MIDI bass's pitch class is the annotated bass (alignment check)
        agree, total = 0.0, 0.0
        for (s, e), label in zip(take.annotation.intervals, take.annotation.labels):
            try:
                root, _, b = data_v8.mir_eval.chord.encode(label)
            except data_v8.mir_eval.chord.InvalidChordException:
                continue
            if root < 0:
                continue
            mid = (s + e) / 2
            k = np.searchsorted(bass_t, mid) - 1
            if k >= 0:
                agree += (e - s) * (bass[k][1] % 12 == (root + b) % 12)
                total += e - s
        rows.append({"key": take.key, "minutes": minutes, "decoded_changes": int(np.count_nonzero(np.diff(path))),
                     "reference_changes": max(len(held) - 1, 0), "inside": len(inside), "inside_with_bass": int(hit.sum()),
                     "inside_on_new_bass": on_bass, "chance": chance, "bass_moves_in_held": held_bass_moves,
                     "bass_agrees_with_label": agree / total if total else None})
    write_measure(rows, skipped)


def tv_fixed_lag_viterbi(posteriors: np.ndarray, Q: np.ndarray, p_self: np.ndarray, prior: np.ndarray,
                        lag: int) -> np.ndarray:
    """online_hmm.fixed_lag_viterbi at one lag with a self-transition probability per frame: the move into
    frame t has p_self[t] on the diagonal and (1 - p_self[t]) * Q off it. Constant p_self gives the usual one."""
    log_b = online_hmm.evidence(posteriors, prior)
    n, T = log_b.shape
    log_q = np.log(Q + online_hmm.EPS)
    eye = np.eye(n, dtype=bool)
    delta = np.empty((T, n))
    psi = np.zeros((T, n), dtype=int)
    delta[0] = log_b[:, 0] - np.log(n)
    for t in range(1, T):
        log_A = np.where(eye, np.log(p_self[t] + online_hmm.EPS), log_q + np.log(1 - p_self[t] + online_hmm.EPS))
        scores = delta[t - 1][:, None] + log_A
        psi[t] = scores.argmax(axis=0)
        delta[t] = scores[psi[t], np.arange(n)] + log_b[:, t]
        delta[t] -= delta[t].max()
    frames = np.arange(T)
    read_at = np.minimum(frames + lag, T - 1)
    state = delta.argmax(axis=1)[read_at]
    for k in range(min(lag, T - 1)):
        step_back = read_at - k > frames
        state = np.where(step_back, psi[np.maximum(read_at - k, 0), state], state)
    return state


def bass_moves(q: np.ndarray) -> np.ndarray:
    """(T,) beta_t = 1 - sum_k q_t(k) q_{t-1}(k): the chance two draws from consecutive bass posteriors differ."""
    beta = np.zeros(q.shape[1])
    beta[1:] = 1 - (q[:, 1:] * q[:, :-1]).sum(axis=0)
    return beta


def slash_labels(path: np.ndarray, q: np.ndarray, names: list[str], half: int = LAG) -> list[str]:
    """Harte label per frame: the chord, with the bass head's note (posterior averaged over t - half .. t + half)
    as its bass when it is a chord tone other than the root."""
    T = len(path)
    c = np.concatenate([np.zeros((q.shape[0], 1)), np.cumsum(q, axis=1)], axis=1)
    lo, hi = np.maximum(np.arange(T) - half, 0), np.minimum(np.arange(T) + half + 1, T)
    bass = ((c[:, hi] - c[:, lo]) / (hi - lo)).argmax(axis=0)
    out = []
    for t, k in enumerate(path):
        root, quality = k % 12, data_v8.VOCABULARIES[r.VOCAB][k // 12]
        degree = TONES[quality].get(int((bass[t] - root) % 12))
        out.append(f"{names[k]}/{degree}" if degree else names[k])
    return out


def head_posteriors(model, song: Song, lookahead: int) -> tuple[np.ndarray, np.ndarray]:
    """(60, T) chord and (12, T) bass posteriors of a whole song, eval mode (model_v6.frame_posteriors' way)."""
    strip = torch.from_numpy(data_v7_2.excerpt(song, 0, 0, song.cqt.shape[1], lookahead))[None, None]
    with torch.no_grad():
        w = model_v6.windows(strip)
        e = torch.cat([model.frame_embeddings(w[i:i + 1024]) for i in range(0, len(w), 1024)])
        h = model.classifier[0](model.gru(e[None])[0])
        chord = torch.softmax(model.classifier[1](h), dim=2)[0].T.numpy()
        q = model.n_qualities
        bass = torch.softmax(model.aux(h)[..., model_v10.N_ROOTS + q:], dim=2)[0].T.numpy()
    return chord, bass


def beatles_val_references(songs: dict) -> dict[str, ChordAnnotation]:
    """{ChoCo key: chordlab annotation} for the Beatles validation songs, matched on the cached targets
    (the share of frames where the chordlab's vocabulary-A target equals the cached one, at least 0.8)."""
    sr, hop = 22050, 2048
    labs = {p: ChordAnnotation.from_lab(p) for album in BEATLES_VAL for p in sorted((BEATLES / album).glob("*.lab"))}
    out = {}
    for key, s in songs.items():
        if s["set"] != "choco":
            continue
        y = s["targets"]
        times = np.arange(len(y)) * hop / sr
        best, best_p = 0.0, None
        for p, a in labs.items():
            x = data_v8.frame_targets(a, times, r.VOCAB)
            m = (x >= 0) | (y >= 0)
            share = float((x[m] == y[m]).mean()) if m.any() else 0.0
            if share > best:
                best, best_p = share, p
        if best >= 0.8:
            out[key] = labs[best_p]
    return out


def step_heads() -> None:
    cfg = r.config()
    sr, hop = cfg["audio"]["sr"], cfg["cqt"]["hop_length"]
    hop_s = hop / sr
    songs = r.load_songs()
    model = model_v10.load(HEADS, r.N_CLASSES, cfg).cpu().eval()
    raw_prior = r.prior_of(HEADS)
    prior = transitions_v8.tempered_prior(raw_prior, cfg["decoder"]["prior_power"])
    Q = r.run_v8.decoder_parts(cfg, r.VOCAB, raw_prior)
    names = data_v8.class_names(r.VOCAB)
    takes = {t.key: t for name in ("rwc", "guitarset") for t in r.val_takes(cfg)[name]}
    refs = {k: t.annotation for k, t in takes.items()}
    refs.update(beatles_val_references(songs))
    chord, bass = {}, {}
    for key, s in songs.items():
        if key in takes:
            song = r.run_v8.load_song(takes[key], cfg, r.VOCAB)
        else:
            cqt = np.load(data_choco.cache_dir(cfg) / f"{key}.npy").astype(np.float32)
            song = Song(key, cqt, s["targets"])
        chord[key], bass[key] = head_posteriors(model, song, cfg["train"]["input_lookahead"])
    saved = r.load_posteriors(HEADS)
    check = max(float(np.abs(chord[k] - saved[k]).max()) for k in saved)
    grid = cfg["decoder"]["p_self_grid"]
    systems = {}                                                   # name -> {key: path}
    systems["network alone"] = {k: p.argmax(axis=0) for k, p in chord.items()}
    for gamma in [0.0, *GAMMAS]:
        for p in grid:
            systems[("lag 8", gamma, p)] = {
                k: tv_fixed_lag_viterbi(chord[k], Q, 1 - (1 - p) * (1 - gamma * bass_moves(bass[k])), prior, LAG)
                for k in chord}
        print(f"gamma {gamma:g} decoded", flush=True)
    scores = {name: r.pooled({k: {("x", 0): path} for k, path in paths_.items()}, songs) for name, paths_ in systems.items()}
    chosen = {"network alone": "network alone"}
    for gamma in [0.0, *GAMMAS]:
        chosen[f"lag 8, gamma {gamma:g}"] = max((("lag 8", gamma, p) for p in grid),
                                                key=lambda n: scores[n][("mean", "x", 0)]["sevenths"])
    midi_bass = {}
    for key in takes:
        if key.startswith("rwc/"):
            notes = midi.bass_notes(midi.read(MIDI_DIR / f"{key.split('/')[1]}.mid"))
            if notes:
                midi_bass[key] = np.array([t for t, _ in midi.bass_line(notes)])
    rows = []
    for label, name in chosen.items():
        paths_ = systems[name]
        row = {"system": label, "p_self": None if name == "network alone" else name[2]}
        for d in (*r.SETS, "mean"):
            row[d] = {m: 100 * scores[name][(d, "x", 0)][m] for m in r.METRICS}
            keys = [k for k in paths_ if d == "mean" or songs[k]["set"] == d]
            row[d]["changes_per_min"] = float(np.mean([np.count_nonzero(np.diff(paths_[k])) / (len(paths_[k]) * hop_s / 60)
                                                       for k in keys]))
        for variant in ("root position", "slash"):
            for d in r.SETS:
                keys = [k for k in paths_ if songs[k]["set"] == d and k in refs]
                for m in ("majmin_inv", "sevenths_inv"):
                    pooled = evaluate.pooled([evaluate.score(refs[k], evaluate.path_to_annotation(
                        np.arange(len(paths_[k])), labels_of(paths_[k], bass[k], names, variant),
                        np.arange(len(paths_[k])) * hop_s, refs[k].intervals.max()), m) for k in keys])
                    row.setdefault(variant, {}).setdefault(d, {})[m] = 100 * pooled.wcsr
                row[variant][d]["songs"] = len(keys)
        inside = with_bass = 0
        minutes = 0.0
        for key, bass_t in midi_bass.items():
            got = changes_inside(paths_[key], held_chords(refs[key]), hop_s)
            inside += len(got)
            with_bass += int(near_any(np.array([t for t, _, _ in got]), bass_t).sum())
            minutes += len(paths_[key]) * hop_s / 60
        row["rwc_inside_per_min"], row["rwc_inside_with_bass_per_min"] = inside / minutes, with_bass / minutes
        rows.append(row)
    write_heads(rows, check, refs, songs)


def labels_of(path: np.ndarray, q: np.ndarray, names: list[str], variant: str) -> list[str]:
    """One label per frame (path_to_annotation reads them by frame index)."""
    return slash_labels(path, q, names) if variant == "slash" else [names[k] for k in path]


def write_heads(rows, check, refs, songs) -> None:
    n_choco = sum(1 for k in refs if k.startswith("choco/"))
    lines = [f"V10's heads model (seed 0), validation, decode only. Chord head posteriors recomputed here match "
             f"the saved ones within {check:.1e}.", "",
             "Majmin / sevenths % (chord changes per minute); p_self chosen per row on the mean sevenths "
             "of the three sets. Last column: decoded changes inside held RWC-P chords per minute (those near a MIDI bass change).", "",
             "| system | p_self | RWC-P | GuitarSet | ChoCo | mean | RWC-P changes inside held chords / min "
             "(near a MIDI bass change) |", "|---|---:|---:|---:|---:|---:|---|"]
    for x in rows:
        cells = [f"{x[d]['majmin']:.2f} / {x[d]['sevenths']:.2f} ({x[d]['changes_per_min']:.0f})" for d in (*r.SETS, "mean")]
        p = "" if x["p_self"] is None else f"{x['p_self']:g}"
        lines.append(f"| {x['system']} | {p} | " + " | ".join(cells)
                     + f" | {x['rwc_inside_per_min']:.2f} ({x['rwc_inside_with_bass_per_min']:.2f}) |")
    lines += ["", f"Inversion-aware scores, majmin_inv / sevenths_inv % (ChoCo: the {n_choco} Beatles validation "
              "songs matched to data/beatles):", "",
              "| system | labels | RWC-P | GuitarSet | ChoCo (Beatles) |", "|---|---|---:|---:|---:|"]
    for x in rows:
        for variant in ("root position", "slash"):
            cells = [f"{x[variant][d]['majmin_inv']:.2f} / {x[variant][d]['sevenths_inv']:.2f}" for d in r.SETS]
            lines.append(f"| {x['system']} | {variant} | " + " | ".join(cells) + " |")
    out = OUT / "bass_heads"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


def write_measure(rows, skipped) -> None:
    m = sum(x["minutes"] for x in rows)
    tot = {k: sum(x[k] for x in rows) for k in ("decoded_changes", "reference_changes", "inside", "inside_with_bass",
                                                "inside_on_new_bass", "bass_moves_in_held")}
    chance = float(np.average([x["chance"] for x in rows], weights=[x["minutes"] for x in rows]))
    agree = float(np.mean([x["bass_agrees_with_label"] for x in rows if x["bass_agrees_with_label"] is not None]))
    lines = [f"RWC-P validation, {len(rows)} songs with a MIDI bass part ({m:.1f} min); left out: {', '.join(skipped) or 'none'}.",
             f"V11 seed 1, fixed-lag Viterbi lag {LAG}, p_self {P_SELF}. Held chord: annotation run of one vocabulary-A "
             f"chord, bass ignored; inside: more than {MARGIN_S} s from its ends; with the bass: a MIDI bass change "
             f"within {WINDOW_S} s.", "",
             "| quantity | count | per minute |", "|---|---:|---:|",
             f"| reference chord changes (vocabulary A, bass ignored) | {tot['reference_changes']} | {tot['reference_changes'] / m:.1f} |",
             f"| decoded chord changes | {tot['decoded_changes']} | {tot['decoded_changes'] / m:.1f} |",
             f"| decoded changes inside a held chord | {tot['inside']} | {tot['inside'] / m:.1f} |",
             f"| of them, with a bass change within {WINDOW_S} s | {tot['inside_with_bass']} | {tot['inside_with_bass'] / m:.1f} |",
             f"| of those, the new chord's root is the new bass note | {tot['inside_on_new_bass']} | {tot['inside_on_new_bass'] / m:.1f} |",
             f"| MIDI bass changes inside held chords | {tot['bass_moves_in_held']} | {tot['bass_moves_in_held'] / m:.1f} |",
             "",
             f"Share of the changes inside a held chord with a bass change near: "
             f"{100 * tot['inside_with_bass'] / max(tot['inside'], 1):.1f} %; chance (share of the time inside held "
             f"chords within {WINDOW_S} s of a bass change): {100 * chance:.1f} %.",
             f"Bass changes inside held chords followed by a decoded change: "
             f"{100 * tot['inside_with_bass'] / max(tot['bass_moves_in_held'], 1):.1f} %.",
             f"Alignment check: the MIDI bass note's pitch class equals the annotated bass on {100 * agree:.1f} % of the "
             "annotated time (song mean)."]
    out = OUT / "bass_measure"
    out.mkdir(parents=True, exist_ok=True)
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["measure", "heads"])
    args = parser.parse_args(argv)
    if args.step == "measure":
        step_measure()
    elif args.step == "heads":
        step_heads()


if __name__ == "__main__":
    main()
