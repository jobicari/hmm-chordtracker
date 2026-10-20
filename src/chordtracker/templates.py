"""Component 3a: the chord vocabulary as binary pitch-class templates.

One template per chord state: a 12-number row with 1 on the chord's pitch classes, 0 elsewhere,
in the same pitch-class order as the chroma rows (0 = C, 1 = C#, ..., 11 = B).
V0 uses the majmin vocabulary: 12 major + 12 minor triads + N (no chord) = 25 states.
Labels are Harte strings with sharps ("D#:maj"), the spelling GuitarSet and mir_eval use.
"""
import numpy as np

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_TRIAD = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # root, major third (+4), fifth (+7)
MINOR_TRIAD = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)  # root, minor third (+3), fifth (+7)
NO_CHORD = "N"


def majmin_labels() -> list[str]:
    """The 25 state labels in template order: C:maj ... B:maj, C:min ... B:min, N.

    State index i is the row i of majmin_templates(): e.g. 3 -> "D#:maj", 21 -> "A:min", 24 -> "N".
    """
    return ([f"{root}:maj" for root in PITCH_CLASSES]
            + [f"{root}:min" for root in PITCH_CLASSES]
            + [NO_CHORD])


def majmin_templates(no_chord_value=0.25) -> np.ndarray:
    """The (25, 12) template matrix; row i is the chord majmin_labels()[i].

    A chord on root r is the root-position triad rolled r steps to the right:
        np.roll(MAJOR_TRIAD, 9) -> 1 on 9, 1 (= 13 mod 12) and 4: A, C#, E = A:maj
    The N row is no_chord_value everywhere (0.25 in V0, as in the librosa example). Its score on a
    max-normalized chroma is 0.25 * (sum of the 12 values), so N only wins when the energy is spread
    over many pitch classes and no triad stands out.
    """
    templates = np.zeros((25, 12))
    for root in range(12):
        templates[root] = np.roll(MAJOR_TRIAD, root)          # rows 0..11: C:maj ... B:maj
        templates[root + 12] = np.roll(MINOR_TRIAD, root)     # rows 12..23: C:min ... B:min
    templates[24] = no_chord_value                            # row 24: N
    return templates
