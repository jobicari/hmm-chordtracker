"""The demo benchmark's arithmetic: a slow step's lag on the frames behind it, and what it suggests."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def bench():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("demo_bench", SCRIPTS / "demo_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_slow_step_delays_the_frames_behind_it_until_the_queue_drains(bench):
    lag = bench.lags(np.array([0.05, 0.3, 0.05, 0.05, 0.05]), 0.1)
    np.testing.assert_allclose(lag, [0.05, 0.3, 0.25, 0.2, 0.15])
    np.testing.assert_allclose(bench.lags(np.full(4, 0.02), 0.1), 0.02)      # always in time: just the step


def test_suggestions_follow_from_the_cpu_per_listener_and_the_worst_lag(bench):
    assert bench.suggestions(0.07, 0.08, 0.15) == (7, 0.2)
    assert bench.suggestions(0.07, 0.01, 0.15) == (7, 0.15)                   # never below the default margin
    assert bench.suggestions(0.9, 0.5, 0.15) == (1, 0.6)                      # at least one session
