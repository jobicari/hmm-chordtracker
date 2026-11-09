"""V13 on validation: the control and played-bass runs against V11, all on the rebuilt (12-bin tuning) features.

    python scripts/v13_eval.py       # -> results/v13/eval/{table.md, results.json}

Systems, two seeds each: V11's frozen recipe as trained (results/v11/pool_kd1_seed<s>, re-scored here on
the rebuilt validation features), the control (results/v13/control_seed<s>: V11's stage B on the rebuilt
cache) and the played-bass run (results/v13/bass_seed<s>). Every validation posterior is computed again
from the current cache and saved as <run>/val_posteriors_v13.npz (bass runs also their bass posteriors).

Per system, V8's four points (the network alone; fixed-lag Viterbi at lag 2, 4, 8 with the p_self of best
mean sevenths, run_v10.best_points), and per point: majmin and sevenths per set and their mean, chord
changes per minute. At lag 8 also: majmin_inv and sevenths_inv (RWC-P, GuitarSet, and of ChoCo the
Beatles validation albums, whose chordlab is in data/beatles), and on RWC-P the bass measure of
scripts/v13_bass.py (decoded changes inside held chords, near a MIDI bass change, re-rooted on the new
bass note). For the bass runs: the slash labels from the bass head, bass-hold decoding over a gamma grid,
and the bass head's frame accuracy against the MIDI bass on RWC-P validation.
"""
import json

import numpy as np
import torch

import run_v10 as r
import v13_bass as vb
from chordtracker import data_choco, data_v5, data_v8, evaluate, midi, model_v13, paths, train_v13, transitions_v8
from chordtracker.data_v5 import Song
from chordtracker.train_v7_2 import song_posteriors

RUNS = {**{f"V11 seed {s}": paths.REPO_ROOT / "results" / "v11" / f"pool_kd1_seed{s}" for s in (0, 1)},
        **{f"control seed {s}": paths.REPO_ROOT / "results" / "v13" / f"control_seed{s}" for s in (0, 1)},
        **{f"bass seed {s}": paths.REPO_ROOT / "results" / "v13" / f"bass_seed{s}" for s in (0, 1)}}
OUT = paths.REPO_ROOT / "results" / "v13" / "eval"
POSTERIORS = "val_posteriors_v13.npz"


def val_songs(cfg: dict, songs: dict) -> dict[str, Song]:
    """Every validation song on the current caches: RWC-P and GuitarSet rebuilt, ChoCo as before."""
    takes = {t.key: t for name in ("rwc", "guitarset") for t in r.val_takes(cfg)[name]}
    out = {}
    for key, s in songs.items():
        if key in takes:
            out[key] = r.run_v8.load_song(takes[key], cfg, r.VOCAB)
        else:
            out[key] = Song(key, np.load(data_choco.cache_dir(cfg) / f"{key}.npy").astype(np.float32), s["targets"])
    return out


def posteriors(run, cfg: dict, songs: dict[str, Song]) -> tuple[dict, dict | None]:
    """({key: (60, T) chord posterior}, {key: (13, T) bass posterior} or None), cached in the run folder."""
    file = run / POSTERIORS
    if not file.exists():
        model = model_v13.load(run, r.N_CLASSES, cfg).cpu().eval()
        la = cfg["train"]["input_lookahead"]
        out = {}
        for key, song in songs.items():
            if isinstance(model, model_v13.ChordCRNNBass):
                chord, bass = vb.head_posteriors_v13(model, song, la)
                out[f"bass|{key}"] = bass.astype(np.float32)
            else:
                chord = song_posteriors(model, song, la)
            out[f"chord|{key}"] = chord.astype(np.float32)
        np.savez(file, **out)
    z = np.load(file)
    chord = {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("chord|")}
    bass = {k.split("|", 1)[1]: z[k] for k in z.files if k.startswith("bass|")}
    return chord, bass or None


def changes_per_min(path: np.ndarray, hop_s: float) -> float:
    return float(np.count_nonzero(np.diff(path))) / (len(path) * hop_s / 60)


def inv_scores(paths_: dict, refs: dict, songs: dict, hop_s: float, labels) -> dict:
    """{set: {majmin_inv, sevenths_inv}} pooled over the songs that have a reference with its bass."""
    out = {}
    for d in r.SETS:
        keys = [k for k in paths_ if songs[k]["set"] == d and k in refs]
        out[d] = {"songs": len(keys)}
        for m in ("majmin_inv", "sevenths_inv"):
            out[d][m] = 100 * evaluate.pooled([evaluate.score(refs[k], evaluate.path_to_annotation(
                np.arange(len(paths_[k])), labels(k), np.arange(len(paths_[k])) * hop_s, refs[k].intervals.max()), m)
                for k in keys]).wcsr
    return out


def bass_measure(paths_: dict, refs: dict, hop_s: float) -> dict:
    """RWC-P validation songs with a MIDI bass: decoded changes inside held chords per minute, those within
    the window of a bass change, and those whose new chord's root is the new bass note."""
    inside = near = rerooted = 0
    minutes = 0.0
    for take in data_v5.rwc_takes("val"):
        notes = midi.bass_notes(midi.read(vb.MIDI_DIR / f"{take.key.split('/')[1]}.mid"))
        if not notes:
            continue
        line = midi.bass_line(notes)
        bass_t, bass_pc = np.array([t for t, _ in line]), np.array([p % 12 for _, p in line])
        got = vb.changes_inside(paths_[take.key], vb.held_chords(refs[take.key]), hop_s)
        hit = vb.near_any(np.array([t for t, _, _ in got]), bass_t)
        inside += len(got)
        near += int(hit.sum())
        for (t, _, after), h in zip(got, hit):
            if h:
                rerooted += int(after % 12 == bass_pc[int(np.argmin(np.abs(bass_t - t)))])
        minutes += len(paths_[take.key]) * hop_s / 60
    return {"inside": inside / minutes, "near_bass": near / minutes, "rerooted": rerooted / minutes}


def bass_accuracy(bass: dict, songs: dict[str, Song], cfg: dict) -> dict:
    """Frame accuracy of the bass head's argmax against the played bass, RWC-P validation songs with a MIDI
    bass: all frames (13 classes) and the frames where a bass note sounds (its pitch class)."""
    right = total = right_p = total_p = 0
    for take in data_v5.rwc_takes("val"):
        notes = midi.bass_notes(midi.read(vb.MIDI_DIR / f"{take.key.split('/')[1]}.mid"))
        if not notes:
            continue
        y = train_v13.played_bass(notes, songs[take.key].cqt.shape[1], cfg["audio"]["sr"], cfg["cqt"]["hop_length"])
        guess = bass[take.key].argmax(axis=0)
        right, total = right + int((guess == y).sum()), total + len(y)
        pitched = y < model_v13.NO_BASS
        right_p, total_p = right_p + int((guess[pitched] == y[pitched]).sum()), total_p + int(pitched.sum())
    return {"all": 100 * right / total, "pitched": 100 * right_p / total_p}


def main() -> None:
    cfg = r.config()
    hop_s = cfg["cqt"]["hop_length"] / cfg["audio"]["sr"]
    songs = r.load_songs()
    feats = val_songs(cfg, songs)
    takes = {t.key: t for name in ("rwc", "guitarset") for t in r.val_takes(cfg)[name]}
    refs = {k: t.annotation for k, t in takes.items()}
    refs.update(vb.beatles_val_references(songs))
    names = data_v8.class_names(r.VOCAB)
    results = {}
    for name, run in RUNS.items():
        chord, bass = posteriors(run, cfg, feats)
        raw_prior = r.prior_of(run)
        decoded = r.decode_all(chord, cfg, raw_prior, cfg["decoder"]["prior_power"])
        sc = r.pooled(decoded, songs)
        chosen = r.best_points(sc)
        res = {"points": {}}
        for point in r.POINT_ORDER:
            p = chosen[point]
            paths_ = {k: v[(point, p)] for k, v in decoded.items()}
            row = {"p_self": p}
            for d in (*r.SETS, "mean"):
                row[d] = {m: 100 * sc[(d, point, p)][m] for m in r.METRICS}
                keys = [k for k in paths_ if d == "mean" or songs[k]["set"] == d]
                row[d]["changes_per_min"] = float(np.mean([changes_per_min(paths_[k], hop_s) for k in keys]))
            res["points"][point] = row
        lag8 = {k: v[("lag 8", chosen["lag 8"])] for k, v in decoded.items()}
        res["inv"] = inv_scores(lag8, refs, songs, hop_s, lambda k: [names[c] for c in lag8[k]])
        res["bass_measure"] = bass_measure(lag8, refs, hop_s)
        if bass is not None:
            q12 = {k: b[:12] for k, b in bass.items()}
            res["inv_slash"] = inv_scores(lag8, refs, songs, hop_s, lambda k: vb.slash_labels(lag8[k], q12[k], names))
            res["bass_accuracy"] = bass_accuracy(bass, feats, cfg)
            Q = r.run_v8.decoder_parts(cfg, r.VOCAB, raw_prior)
            prior = transitions_v8.tempered_prior(raw_prior, cfg["decoder"]["prior_power"])
            p = chosen["lag 8"]
            res["bass_hold"] = {}
            for gamma in vb.GAMMAS:
                held = {k: vb.tv_fixed_lag_viterbi(chord[k], Q, 1 - (1 - p) * (1 - gamma * vb.bass_moves(bass[k])),
                                                   prior, vb.LAG) for k in chord}
                s = r.pooled({k: {("x", 0): v} for k, v in held.items()}, songs)
                res["bass_hold"][gamma] = {"mean": {m: 100 * s[("mean", "x", 0)][m] for m in r.METRICS},
                                           "changes_per_min": float(np.mean([changes_per_min(v, hop_s) for v in held.values()])),
                                           "bass_measure": bass_measure(held, refs, hop_s)}
        results[name] = res
        print(f"{name} done", flush=True)
    write(results)


def write(results: dict) -> None:
    lines = ["Validation, rebuilt features (12-bin tuning). Cells: majmin / sevenths % (changes per minute).", "",
             "| system | point | p_self | RWC-P | GuitarSet | ChoCo | mean |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, res in results.items():
        for point, row in res["points"].items():
            cells = [f"{row[d]['majmin']:.2f} / {row[d]['sevenths']:.2f} ({row[d]['changes_per_min']:.0f})"
                     for d in (*r.SETS, "mean")]
            p = "" if row["p_self"] is None else f"{row['p_self']:g}"
            lines.append(f"| {name} | {r.POINT_NAMES[point]} | {p} | " + " | ".join(cells) + " |")
    lines += ["", "Lag 8: majmin_inv / sevenths_inv % (ChoCo: Beatles validation songs), and on RWC-P the decoded "
              "changes inside held chords per minute (near a MIDI bass change; re-rooted on the new bass note).", "",
              "| system | labels | RWC-P | GuitarSet | ChoCo (Beatles) | inside | near bass | re-rooted |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, res in results.items():
        bm = res["bass_measure"]
        for label, key in (("root position", "inv"), ("slash", "inv_slash")):
            if key not in res:
                continue
            cells = [f"{res[key][d]['majmin_inv']:.2f} / {res[key][d]['sevenths_inv']:.2f}" for d in r.SETS]
            tail = f"{bm['inside']:.2f} | {bm['near_bass']:.2f} | {bm['rerooted']:.2f}" if key == "inv" else " |  | "
            lines.append(f"| {name} | {label} | " + " | ".join(cells) + f" | {tail} |")
    bass_runs = {n: res for n, res in results.items() if "bass_hold" in res}
    if bass_runs:
        lines += ["", "Bass runs: bass head frame accuracy against the MIDI bass on RWC-P validation, all frames "
                  "(13 classes) / frames with a bass note; bass-hold decoding at lag 8 (mean majmin / sevenths, "
                  "changes per minute, RWC-P inside / near bass).", "",
                  "| system | bass accuracy | gamma | mean | changes / min | inside | near bass |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for name, res in bass_runs.items():
            acc = f"{res['bass_accuracy']['all']:.1f} / {res['bass_accuracy']['pitched']:.1f}"
            for gamma, h in res["bass_hold"].items():
                lines.append(f"| {name} | {acc} | {gamma:g} | {h['mean']['majmin']:.2f} / {h['mean']['sevenths']:.2f} | "
                             f"{h['changes_per_min']:.0f} | {h['bass_measure']['inside']:.2f} | "
                             f"{h['bass_measure']['near_bass']:.2f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
