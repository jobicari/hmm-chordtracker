"""Behaviour of the frozen V8 test run: it needs --freeze and runs only once."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v8_test  # noqa: E402


def test_it_needs_freeze():
    with pytest.raises(SystemExit):
        v8_test.parse_args([])


def test_it_refuses_a_second_run(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="scored once"):
        v8_test.main(["--freeze", "--out", str(tmp_path)])
