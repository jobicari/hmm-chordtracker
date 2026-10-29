"""Component 27 (V7.3): a hold rule against flicker, with a look-ahead of m - 1 frames.

The network alone changes chord far more often than the music (about 60 times per minute at V7.2's
0.28 s point, against 28 to 31). The hold rule keeps the current chord until a new one has been the
network's most likely chord for m frames in a row:

    label of frame t = c   if the argmax is c on every frame t .. t + m - 1
                       = the label of frame t - 1 otherwise (frame 0: its own argmax)

It reads m - 1 frames past frame t, so with delayed playback it costs m - 1 frames of total look-ahead,
like a decoder lag of m - 1. m = 1 is the network alone.
"""
import numpy as np


def hold(argmax: np.ndarray, m: int) -> np.ndarray:
    """(T,) per-frame argmax -> (T,) held labels. Example, m = 2: [0, 1, 0, 2, 2, 2] -> [0, 0, 0, 2, 2, 2]."""
    out = np.empty_like(argmax)
    for t in range(len(argmax)):
        window = argmax[t:t + m]
        if t == 0 or (len(window) == m and (window == window[0]).all()):
            out[t] = argmax[t]
        else:
            out[t] = out[t - 1]
    return out
