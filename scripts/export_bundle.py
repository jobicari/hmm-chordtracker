"""Export the frozen V15 as a bundle for the standalone macOS app (chordtracker.bundle, reports/bundle.md).

Runs in the export environment, not the project's: coremltools 9.0 converts torch up to 2.7.0, and the
project's torch is newer. Every other package is the demo image's (requirements-export.txt):

    python3.12 -m venv .venv-export && .venv-export/bin/pip install -r requirements-export.txt
    PYTHONPATH=src .venv-export/bin/python scripts/export_bundle.py --out <folder>

Writes <folder>/v15/:

    README.md               reports/bundle.md: every stage of the live chain as a port computes it
    system.json, decoder/   the settings, labels, points and decoder arrays (bundle.describe)
    ChordStep.mlpackage     the causal CRNN, one frame per call (bundle.FrameStep), float32, Core ML
    golden/<case>/          two recordings through the live chain, every stage kept (bundle.capture):
                              funk                the demo's GuitarSet validation take 00_Funk2-108-Eb_comp
                              jazz_flat_silence   02_Jazz2-110-Bb_comp's first 24 s, 35 cents flat (outside
                                                  the +-25 cents a 24-bin estimate reads), after 1.5 s of
                                                  digital silence (the tuning starts at the first audible frame)

Then checks the bundle with Core ML itself: every golden case's columns through ChordStep.mlpackage and the
bundle's decoder arrays, against the golden posteriors and labels (bundle.check). Core ML compiles the model
in the user's temporary folder, so this runs in a Terminal; a sandbox without that folder stops it.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

from chordtracker import bundle, demo, model_v6, paths
from chordtracker.data_v5 import PATCH_BINS, PATCH_FRAMES
from chordtracker.play_along import SILENCE_DBFS, SYSTEMS, load_chain, system_labels

sys.path.insert(0, str(Path(__file__).parent))
import demo_export  # noqa: E402
import run_v5  # noqa: E402

SYSTEM = "v15"
BLOCK = 2048                    # samples per step of the golden runs (any block size gives the same labels)
CASES = {                       # name: GuitarSet validation take, pitch shift (cents), silence before (s), length (s)
    "funk": ("00_Funk2-108-Eb_comp", 0, 0.0, None),
    "jazz_flat_silence": ("02_Jazz2-110-Bb_comp", -35, 1.5, 24.0),
}
README = paths.REPO_ROOT / "reports" / "bundle.md"


def git_source() -> dict:
    """The repository and commit the bundle is exported from, and whether the tree had changes."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"repository": "jobicari/hmm-chordtracker", "commit": git("rev-parse", "HEAD"),
            "uncommittedChanges": bool(git("status", "--porcelain", "--untracked-files=no"))}


def case_audio(take: str, cents: int, silence_s: float, seconds: float | None, sr: int) -> np.ndarray:
    """The take at sr, mono (librosa.load), cut to `seconds`, shifted by `cents`, after silence_s of zeros."""
    y, _ = librosa.load(str(demo_export.take_of(take).audio), sr=sr, mono=True, duration=seconds)
    if cents:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=cents / 100)
    return np.concatenate([np.zeros(int(silence_s * sr), dtype=np.float32), y.astype(np.float32)])


def convert(model, out: Path) -> None:
    """bundle.FrameStep of the model as an ML program (float32, macOS 14), inputs window and hidden, outputs
    posteriors and hidden_out."""
    import coremltools as ct

    window, hidden = torch.zeros(1, 1, PATCH_BINS, PATCH_FRAMES), torch.zeros(1, 1, model_v6.HIDDEN)
    with torch.no_grad():
        traced = torch.jit.trace(bundle.FrameStep(model).eval(), (window, hidden))
    ml = ct.convert(traced, inputs=[ct.TensorType(name="window", shape=window.shape),
                                    ct.TensorType(name="hidden", shape=hidden.shape)],
                    outputs=[ct.TensorType(name="posteriors"), ct.TensorType(name="hidden_out")],
                    convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
                    minimum_deployment_target=ct.target.macOS14, skip_model_load=True)
    ml.short_description = ("V15 chord network (hmm-chordtracker), one frame: P(class | frame t) from the "
                            "log-CQT window ending at frame t + 2 and the GRU state")
    ml.input_description["window"] = "log-CQT rows 12..155 (C2..B7), 15 columns, oldest first"
    ml.input_description["hidden"] = "GRU state, zeros before the first frame"
    ml.output_description["posteriors"] = "97 class probabilities in decode order (system.json labels, N last)"
    ml.output_description["hidden_out"] = "GRU state for the next frame"
    ml.save(str(out))


def export(out: Path) -> Path:
    """Write the bundle under out/v15/ and return that folder."""
    system = SYSTEMS[SYSTEM]
    cfg = run_v5.load_config(paths.REPO_ROOT / system.config)
    model, prior, Q = load_chain(SYSTEM, cfg, paths.REPO_ROOT / system.run)
    labels, points = system_labels(SYSTEM, cfg), demo.MODELS[SYSTEM]
    folder = out / SYSTEM
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    description, arrays = bundle.describe(SYSTEM, cfg, system.run, prior, Q, labels, points, git_source())
    for name, array in arrays.items():
        bundle.write_array(folder, name, array, "f64")
    convert(model, folder / "ChordStep.mlpackage")
    sr = cfg["audio"]["sr"]
    cases = {}
    for name, (take, cents, silence_s, seconds) in CASES.items():
        y = case_audio(take, cents, silence_s, seconds, sr)
        scalars, case_arrays = bundle.capture(y, cfg, model, prior, Q, points, len(labels) - 1, SILENCE_DBFS, BLOCK)
        about = {"source": {"dataset": "GuitarSet", "take": take, "split": "validation", "license": "CC BY 4.0",
                            "url": "https://guitarset.weebly.com/"},
                 "audio": {"pitchShiftCents": cents, "silenceBeforeSeconds": silence_s, "seconds": seconds,
                           "loaded": "librosa.load(sr=22050, mono=True), then librosa.effects.pitch_shift"}}
        bundle.write_case(folder, name, scalars, case_arrays, about)
        cases[name] = scalars["frames"]
        print(f"{name}: {scalars['samples'] / sr:.1f} s, {scalars['frames']} frames, tuning {scalars['tuning']['semitones']}"
              f" semitones, streamed vs recomputed log-CQT {scalars['streamedVsRecomputedLogcqt']:.2e}", flush=True)
    description["golden"] = sorted(cases)
    (folder / "system.json").write_text(json.dumps(description, indent=1))
    shutil.copyfile(README, folder / "README.md")
    return folder


def coreml_predict(folder: Path):
    """predict(window, hidden) -> (posteriors, hidden_out) of the bundle's Core ML model, on the CPU."""
    import coremltools as ct

    ml = ct.models.MLModel(str(folder / "ChordStep.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)

    def predict(window, hidden):
        out = ml.predict({"window": window, "hidden": hidden})
        return out["posteriors"], out["hidden_out"]
    return predict


def report(results: dict) -> bool:
    """Print the check; True if every label agrees."""
    ok = True
    for case, r in results.items():
        agree = ", ".join(f"{p} {100 * share:.2f} %" for p, share in r["labelsAgree"].items())
        print(f"check {case}: posteriors max abs diff {r['posteriorsMaxAbsDiff']:.2e}; labels agree: {agree}")
        ok &= all(share == 1.0 for share in r["labelsAgree"].values())
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="the bundle goes in OUT/v15/")
    args = parser.parse_args()
    folder = export(args.out)
    if not report(bundle.check(folder, coreml_predict(folder))):
        raise SystemExit("the Core ML model does not give the golden labels")
    print(f"bundle written to {folder}")


if __name__ == "__main__":
    main()
