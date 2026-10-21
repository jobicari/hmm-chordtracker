"""Tests for the command line of scripts/run_v0.py: the guards that keep the test split from being swept."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v0.py"


@pytest.fixture(scope="module")
def run_v0():
    """Import scripts/run_v0.py as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("run_v0", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_val_sweeps_without_p_self(run_v0):
    """--split val is accepted on its own and keeps p_self empty (the grid comes from the config)."""
    args = run_v0.parse_args(["--split", "val"])
    assert args.split == "val" and args.p_self is None


def test_test_split_needs_p_self(run_v0):
    """--split test without --p-self exits with an error; with --p-self 0.99 it is accepted."""
    with pytest.raises(SystemExit):
        run_v0.parse_args(["--split", "test"])
    assert run_v0.parse_args(["--split", "test", "--p-self", "0.99"]).p_self == 0.99


def test_val_refuses_p_self(run_v0):
    """--split val --p-self 0.9 exits with an error: val always sweeps the whole grid."""
    with pytest.raises(SystemExit):
        run_v0.parse_args(["--split", "val", "--p-self", "0.9"])
