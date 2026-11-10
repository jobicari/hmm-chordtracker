"""The play-along script's model switch: which config, run and silence gate each choice gives."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def script():
    pytest.importorskip("sounddevice")
    pytest.importorskip("tkinter")
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("play_along_script", SCRIPTS / "play_along.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v11_by_default_with_the_gate(script):
    from chordtracker import paths, play_along as pa
    cfg, run, gate = script.chain_settings(script.parse_args(["--file", "x.wav"]))
    assert run == paths.REPO_ROOT / "results" / "v11" / "pool_kd1_seed1" and gate == pa.SILENCE_DBFS
    assert cfg["version"] == "v11"


def test_v14_without_the_gate_unless_asked(script, tmp_path):
    from chordtracker import paths, play_along as pa
    cfg, run, gate = script.chain_settings(script.parse_args(["--model", "v14", "--file", "x.wav"]))
    assert cfg["version"] == "v14" and run == paths.REPO_ROOT / "results" / "v14" / "n_seed0" and gate is None
    _, run, gate = script.chain_settings(script.parse_args(["--model", "v14", "--silence-gate", "--run", str(tmp_path),
                                                            "--input-device", "VB-Cable A"]))
    assert run == tmp_path and gate == pa.SILENCE_DBFS
    assert script.chain_settings(script.parse_args(["--no-silence-gate", "--file", "x.wav"]))[2] is None


def test_the_input_is_required(script):
    with pytest.raises(SystemExit):
        script.parse_args(["--model", "v14"])
    with pytest.raises(SystemExit):
        script.parse_args(["--input-device", "A", "--dry-run", "5"])
    assert script.parse_args(["--list-devices"]).list_devices


def test_v15_without_the_gate(script):
    from chordtracker import paths
    cfg, run, gate = script.chain_settings(script.parse_args(["--model", "v15", "--file", "x.wav"]))
    assert cfg["version"] == "v15" and run == paths.REPO_ROOT / "results" / "v15" / "ft_seed0" and gate is None
    assert cfg["audio"]["sr"] == 22050 and cfg["vocab_plus"]["new"] == ["dim", "aug", "hdim7"]   # v2 + v14 + v15
