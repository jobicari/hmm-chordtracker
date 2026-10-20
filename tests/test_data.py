"""Tests for chordtracker.data: ChordAnnotation, GuitarSetAdapter, split rules and prepare().

Unit tests build tiny fake datasets in pytest's tmp_path and run in about a second.
The test marked `slow` reads the real GuitarSet in data/guitarset: run it with `pytest -m slow`.
"""
from pathlib import Path

import jams
import numpy as np
import pytest

from chordtracker.data import data as d

GUITARSET = Path(__file__).resolve().parents[1] / "data" / "guitarset"

LEAD = [(0.0, 4.0, "C:maj"), (4.0, 8.0, "A:min"), (8.0, 12.0, "G:7")]
PLAYED = [(0.0, 4.0, "C:maj7"), (4.0, 8.0, "A:min7"), (8.0, 12.0, "G:7")]
NAMES = ["00_BN1-129-Eb_comp", "00_BN1-129-Eb_solo", "01_Funk2-108-Bb_comp"]
GS_GROUPS = [f"{style}{prog}" for style in ("BN", "Funk", "Jazz", "Rock", "SS") for prog in (1, 2, 3)]


# ------------------------------------------------------------------------ helpers

def make_jams(path, instructed, performed, duration=12.0):
    """JAMS file with a tempo annotation (must be ignored), then lead-sheet and performed chords."""
    jam = jams.JAMS()
    jam.file_metadata.duration = duration
    tempo = jams.Annotation(namespace="tempo")
    tempo.append(time=0.0, duration=duration, value=120.0, confidence=1.0)
    jam.annotations.append(tempo)
    for chords in (instructed, performed):
        ann = jams.Annotation(namespace="chord")
        for start, end, label in chords:
            ann.append(time=start, duration=end - start, value=label, confidence=1.0)
        jam.annotations.append(ann)
    jam.save(str(path))


def make_guitarset(root, names, with_audio=True):
    """Fake GuitarSet layout: root/annotation/*.jams and root/audio_mono-mic_comp/*_comp_mic.wav."""
    (root / "annotation").mkdir(parents=True)
    (root / "audio_mono-mic_comp").mkdir()
    for name in names:
        make_jams(root / "annotation" / f"{name}.jams", LEAD, PLAYED)
        if with_audio and name.endswith("_comp"):
            (root / "audio_mono-mic_comp" / f"{name}_mic.wav").touch()    # contents are never read here
    return root


def annotation():
    """Three chords with a gap between 8 and 9 s and nothing before 0.5 s: C:maj, A:min, G:maj."""
    return d.ChordAnnotation(np.array([[0.5, 4.0], [4.0, 8.0], [9.0, 12.0]]), ["C:maj", "A:min", "G:maj"])


class ListAdapter(d.DatasetAdapter):
    """Smallest concrete adapter: yields the RawTakes it is given and keeps the default split."""

    def __init__(self, takes):
        self.takes = takes

    def raw_takes(self):
        yield from self.takes


def raw(song_id, group, folder):
    """RawTake with two chords (C:maj 0-2 s, G:maj 2-4 s) and an audio path that is never opened."""
    chords = d.ChordAnnotation(np.array([[0.0, 2.0], [2.0, 4.0]]), ["C:maj", "G:maj"])
    return d.RawTake(song_id, group, folder / f"{song_id}.wav", chords)


@pytest.fixture
def gs(tmp_path):
    """GuitarSetAdapter on an empty fake layout: enough to call split_groups(), which reads no files."""
    return d.GuitarSetAdapter(make_guitarset(tmp_path, []))


@pytest.fixture
def prepared(tmp_path):
    """prepare() run on 10 takes in 5 groups (g0..g4, two takes each), seed 0.

    Returns (out_dir, takes): the folder holding labs/, manifest.csv and split.json, and the TakeRecords.
    """
    takes = [raw(f"g{i}_take{k}", f"g{i}", tmp_path) for i in range(5) for k in range(2)]
    out = tmp_path / "out"
    return out, d.prepare(ListAdapter(takes), out, seed=0)


# ------------------------------------------------------------------------ ChordAnnotation

def test_annotation_rejects_mismatched_lengths():
    """1 interval for 2 labels raises ValueError when the annotation is created, not later when it is used."""
    with pytest.raises(ValueError):
        d.ChordAnnotation(np.array([[0.0, 1.0]]), ["C:maj", "G:maj"])


def test_annotation_lab_round_trip(tmp_path):
    """to_lab then from_lab gives back the same intervals and labels, gap included."""
    ann = annotation()
    ann.to_lab(tmp_path / "x.lab")
    back = d.ChordAnnotation.from_lab(tmp_path / "x.lab")
    np.testing.assert_allclose(back.intervals, ann.intervals)
    assert back.labels == ann.labels


def test_annotation_at_boundaries_gaps_and_outside():
    """at() returns the label at each time, and "N" before the first chord, in gaps and from the end on.

    A boundary time belongs to the chord that starts there (4.0 s -> A:min), and the final end
    time is already outside (12.0 s -> N): segments are half-open, [start, end).
    """
    # before the first start, a start, a boundary (belongs to the new chord), in the 8-9 s gap,
    # just inside the last segment, exactly at the final end, past the end
    times = np.array([0.2, 0.5, 4.0, 8.5, 11.99, 12.0, 15.0])
    assert annotation().at(times) == ["N", "C:maj", "A:min", "N", "G:maj", "N", "N"]


def test_annotation_iterates_over_segments():
    """Iterating gives ((start, end), label) pairs: the second one is ((4.0, 8.0), "A:min")."""
    assert list(annotation())[1] == ((4.0, 8.0), "A:min")


def test_annotation_from_jams_picks_the_requested_chord_annotation(tmp_path):
    """index counts chord annotations only: the tempo annotation in front is skipped.

    index=0 gives the lead-sheet chords (C:maj, A:min, G:7), index=1 the performed ones
    (C:maj7, A:min7, G:7), with the JAMS (time, duration) turned into [start, end] intervals.
    """
    make_jams(tmp_path / "x.jams", LEAD, PLAYED)
    lead = d.ChordAnnotation.from_jams(tmp_path / "x.jams", index=0)
    played = d.ChordAnnotation.from_jams(tmp_path / "x.jams", index=1)
    assert lead.labels == ["C:maj", "A:min", "G:7"]
    assert played.labels == ["C:maj7", "A:min7", "G:7"]
    np.testing.assert_allclose(lead.intervals, [[0, 4], [4, 8], [8, 12]])


# ------------------------------------------------------------------------ adapters

def test_adapter_without_raw_takes_cannot_be_created():
    """A subclass that forgets raw_takes() raises TypeError at creation, thanks to @abstractmethod."""
    class Incomplete(d.DatasetAdapter):
        pass

    with pytest.raises(TypeError):
        Incomplete()


def test_guitarset_missing_folder_fails_early(tmp_path):
    """An empty root (no annotation/ or audio folder) raises FileNotFoundError in __init__, before any take is read."""
    with pytest.raises(FileNotFoundError):
        d.GuitarSetAdapter(tmp_path)


def test_guitarset_raw_takes_keeps_comp_and_parses_group(tmp_path):
    """Solo takes are skipped, the group is parsed from the file name and lead-sheet labels are the default.

    "00_BN1-129-Eb_comp" -> group "BN1", "01_Funk2-108-Bb_comp" -> group "Funk2"; the solo take is dropped.
    """
    takes = list(d.GuitarSetAdapter(make_guitarset(tmp_path, NAMES)).raw_takes())
    assert [t.song_id for t in takes] == ["00_BN1-129-Eb_comp", "01_Funk2-108-Bb_comp"]
    assert [t.group for t in takes] == ["BN1", "Funk2"]
    assert all(t.audio.is_file() for t in takes)
    assert takes[0].chords.labels == ["C:maj", "A:min", "G:7"]


def test_guitarset_performed_labels(tmp_path):
    """labels="performed" switches to the second chord annotation (C:maj7, A:min7, G:7)."""
    adapter = d.GuitarSetAdapter(make_guitarset(tmp_path, NAMES), labels="performed")
    assert next(iter(adapter.raw_takes())).chords.labels == ["C:maj7", "A:min7", "G:7"]


def test_guitarset_missing_audio_raises(tmp_path):
    """A comp take whose .wav is missing raises FileNotFoundError instead of being skipped silently."""
    adapter = d.GuitarSetAdapter(make_guitarset(tmp_path, NAMES, with_audio=False))
    with pytest.raises(FileNotFoundError):
        list(adapter.raw_takes())


# ------------------------------------------------------------------------ split rules

def test_default_split_assigns_every_group_once():
    """The default rule puts every group in exactly one split, 60/20/20, independent of input order.

    10 groups -> 6 train, 2 val, 2 test. Reversing the input list gives the same split, because
    the groups are sorted before the seeded shuffle.
    """
    groups = [f"g{i}" for i in range(10)]
    split = ListAdapter([]).split_groups(groups, seed=0)
    assert set(split) == set(groups)
    values = list(split.values())
    assert (values.count("train"), values.count("val"), values.count("test")) == (6, 2, 2)
    assert split == ListAdapter([]).split_groups(groups[::-1], seed=0)     # input order doesn't matter


def test_guitarset_split_one_progression_per_style_in_each_split(gs):
    """GuitarSet rule: each of train/val/test gets one progression of every style (BN, Funk, Jazz, Rock, SS)."""
    split = gs.split_groups(GS_GROUPS, seed=0)
    assert set(split) == set(GS_GROUPS)
    for name in d.SPLITS:
        styles = sorted(g.rstrip("0123456789") for g, s in split.items() if s == name)
        assert styles == ["BN", "Funk", "Jazz", "Rock", "SS"]


def test_guitarset_split_is_reproducible(gs):
    """Same seed gives the same split whatever the input order, and seed 0 gives the split already in use.

    Train groups for seed 0 are BN1, Funk3, Jazz1, Rock1 and SS1, as in splits/guitarset.json.
    """
    split = gs.split_groups(GS_GROUPS, seed=0)
    assert split == gs.split_groups(GS_GROUPS[::-1], seed=0)
    # the seed-0 split used so far; if this changes, every reported score changes with it
    assert sorted(g for g, s in split.items() if s == "train") == ["BN1", "Funk3", "Jazz1", "Rock1", "SS1"]


def test_guitarset_split_rejects_incomplete_style(gs):
    """A style with 2 progressions instead of 3 cannot be spread over 3 splits, so it raises ValueError."""
    with pytest.raises(ValueError):
        gs.split_groups(["BN1", "BN2"], seed=0)


# ------------------------------------------------------------------------ prepare

def test_prepare_writes_one_lab_per_take(prepared):
    """prepare() writes one .lab per take (10 here), and each reads back as the take's chords."""
    out, takes = prepared
    assert len(takes) == 10
    for t in takes:
        assert d.ChordAnnotation.from_lab(t.lab).labels == ["C:maj", "G:maj"]


def test_prepare_manifest_round_trip(prepared):
    """load_manifest() on the written manifest.csv gives back the same records, with the relative paths resolved."""
    out, takes = prepared
    loaded = d.load_manifest(out / "manifest.csv")
    key = lambda t: (t.song_id, t.group, Path(t.audio).resolve(), Path(t.lab).resolve())
    assert [key(t) for t in loaded] == [key(t) for t in takes]


def test_prepare_split_keeps_groups_together(prepared):
    """Every take is in exactly one split, and both takes of a group are in the same one."""
    out, takes = prepared
    split = d.load_split(out / "split.json")
    assert sorted(sid for ids in split.values() for sid in ids) == sorted(t.song_id for t in takes)
    home = {sid: name for name, ids in split.items() for sid in ids}
    for t in takes:
        assert home[t.song_id] == home[f"{t.group}_take0"]


def test_prepare_writes_split_to_split_path(tmp_path):
    """With split_path given, the split goes there (e.g. splits/toy.json) and no split.json is left in out_dir."""
    takes = [raw(f"g{i}_take0", f"g{i}", tmp_path) for i in range(5)]
    split_path = tmp_path / "splits" / "toy.json"
    d.prepare(ListAdapter(takes), tmp_path / "out", seed=0, split_path=split_path)
    assert sorted(sid for ids in d.load_split(split_path).values() for sid in ids) == [t.song_id for t in takes]
    assert not (tmp_path / "out" / "split.json").exists()


def test_prepare_refuses_to_replace_a_different_split(tmp_path):
    """Running prepare() again with the same seed is fine; a different seed raises instead of overwriting.

    This protects splits/guitarset.json: changing the official split has to be done on purpose.
    """
    takes = [raw(f"g{i}_take0", f"g{i}", tmp_path) for i in range(5)]
    split_path = tmp_path / "split.json"
    d.prepare(ListAdapter(takes), tmp_path / "out", seed=0, split_path=split_path)
    d.prepare(ListAdapter(takes), tmp_path / "out", seed=0, split_path=split_path)    # same split: fine
    with pytest.raises(ValueError):
        d.prepare(ListAdapter(takes), tmp_path / "out", seed=1, split_path=split_path)


# ------------------------------------------------------------------------ real data

@pytest.mark.slow
@pytest.mark.skipif(not GUITARSET.is_dir(), reason="GuitarSet not found in data/guitarset")
def test_real_guitarset_counts():
    """On the real GuitarSet (data/guitarset, run with `pytest -m slow`): the numbers the project relies on.

    180 comp takes in 15 groups of 12, first take 00_BN1-129-Eb_comp starting D#:maj, G#:maj
    (Eb written as D# by the annotation), and the seed-0 split gives 60/60/60 takes.
    """
    adapter = d.GuitarSetAdapter(GUITARSET)
    takes = list(adapter.raw_takes())
    assert len(takes) == 180
    assert takes[0].song_id == "00_BN1-129-Eb_comp"
    assert takes[0].chords.labels[:2] == ["D#:maj", "G#:maj"]
    groups = {t.group for t in takes}
    assert len(groups) == 15
    assert all(sum(t.group == g for t in takes) == 12 for g in groups)
    split = adapter.split_groups(groups, seed=0)
    assert [sum(split[t.group] == s for t in takes) for s in d.SPLITS] == [60, 60, 60]
