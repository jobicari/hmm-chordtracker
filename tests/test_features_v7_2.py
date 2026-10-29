"""The causal features on the C2 grid (fmin F#1): still no column reads the future, and the low notes wait less."""
import numpy as np

from chordtracker import features_v7_1 as f
from test_features_v7_1 import CFG, HOP, SR, synthetic

C2 = {**CFG, "cqt": {**CFG["cqt"], "fmin": "F#1"}}


def test_no_column_reads_the_future_on_the_c2_grid():
    y = synthetic()
    cut = int(12.5 * SR)
    full, part = f.causal_log_cqt_of_audio(y, C2), f.causal_log_cqt_of_audio(y[:cut], C2)
    final = cut // HOP - 1 - f.LOOKAHEAD_FRAMES
    np.testing.assert_allclose(part[:, :final], full[:, :final], atol=1e-3)


def test_the_lowest_note_the_network_sees_waits_half_as_long():
    assert f.bin_delays(CFG)[12] == 6            # row 12 of the F#0 grid is C1
    assert f.bin_delays(C2)[12] == 3             # row 12 of the F#1 grid is C2
