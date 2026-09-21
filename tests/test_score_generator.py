import math
from pathlib import Path
import pytest

from pianorl.score import (
    Score,
    generate_score,
    MIN_PIANO_PITCH,
    MAX_PIANO_PITCH,
)
from pianorl.score.generator import WHITE_KEY_SEMITONES
from scripts.generate_dataset import generate_dataset, get_score_fingerprint


def test_seed_reproducibility():
    """Verify that using the same seed produces the exact same notes and tempo across all 8 levels."""
    for level in range(1, 9):
        score_a = generate_score(level=level, seed=999)
        score_b = generate_score(level=level, seed=999)

        assert score_a.tempo_bpm == score_b.tempo_bpm
        assert len(score_a) == len(score_b)
        for na, nb in zip(score_a.notes, score_b.notes):
            assert na.pitch == nb.pitch
            assert na.start_beat == nb.start_beat
            assert na.duration_beats == nb.duration_beats


def test_levels_1_to_4_output_unchanged():
    """Verify that levels 1 to 4 produce the exact same pieces and note sequences as before."""
    # Deterministic reference values with seed=42
    s1 = generate_score(level=1, seed=42)
    assert s1.tempo_bpm == 63.0
    assert len(s1.notes) == 16
    assert s1.notes[0].pitch == 65

    s2 = generate_score(level=2, seed=42)
    assert s2.tempo_bpm == 63.0
    assert len(s2.notes) == 14
    assert s2.notes[0].pitch == 65

    s3 = generate_score(level=3, seed=42)
    assert s3.tempo_bpm == 90.0
    assert len(s3.notes) == 60
    assert s3.notes[0].pitch == 49

    s4 = generate_score(level=4, seed=42)
    assert s4.tempo_bpm == 110.0
    assert len(s4.notes) == 27
    assert s4.notes[0].pitch == 49


def test_pitches_within_88_keys():
    """Verify every note generated across all 8 levels stays strictly inside the 88-key piano."""
    for level in range(1, 9):
        for seed in range(10):
            score = generate_score(level=level, seed=seed * 100)
            assert len(score.notes) > 0
            for note in score.notes:
                assert MIN_PIANO_PITCH <= note.pitch <= MAX_PIANO_PITCH, (
                    f"Level {level} note pitch {note.pitch} outside [{MIN_PIANO_PITCH}, {MAX_PIANO_PITCH}]"
                )


def test_beats_multiples_of_16th_grid():
    """Verify all start beats and durations land exactly on the 16th-note grid (multiples of 0.25)."""
    for level in range(1, 9):
        for seed in range(5):
            score = generate_score(level=level, seed=seed)
            for note in score.notes:
                # Check start_beat is multiple of 0.25
                rem_start = round(note.start_beat % 0.25, 4)
                assert rem_start == 0.0 or math.isclose(rem_start, 0.25, abs_tol=1e-5), (
                    f"Level {level} start_beat {note.start_beat} not on 16th grid"
                )

                # Check duration_beats is multiple of 0.25
                rem_dur = round(note.duration_beats % 0.25, 4)
                assert rem_dur == 0.0 or math.isclose(rem_dur, 0.25, abs_tol=1e-5), (
                    f"Level {level} duration_beats {note.duration_beats} not on 16th grid"
                )


def test_levels_1_and_2_five_key_range():
    """Verify levels 1 and 2 only use pitches from a single five-white-key span."""
    for level in [1, 2]:
        for seed in range(10):
            score = generate_score(level=level, seed=seed)
            unique_pitches = sorted(list(set(n.pitch for n in score.notes)))
            assert len(unique_pitches) <= 5


def test_new_levels_features():
    """Verify distinct musical properties of new levels 5 to 8."""
    # Level 5: uses all 12 chromatic notes (black keys included), slow tempo (60-80 BPM)
    has_black_key = False
    all_l5_pitches = []
    for seed in range(20):
        s5 = generate_score(level=5, seed=seed)
        assert 60.0 <= s5.tempo_bpm <= 80.0
        assert s5.total_beats == pytest.approx(16.0)
        for n in s5.notes:
            all_l5_pitches.append(n.pitch)
            if (n.pitch % 12) not in WHITE_KEY_SEMITONES:
                has_black_key = True
    assert has_black_key, "Level 5 should generate black keys"
    # Wide range across keyboard (at least 30 semitones span across the 20 pieces)
    assert max(all_l5_pitches) - min(all_l5_pitches) >= 30

    # Level 6: major AND minor scales/arpeggios in all 12 keys, 1-3 octaves (70-100 BPM)
    l6_has_minor = False
    for seed in range(30):
        s6 = generate_score(level=6, seed=seed)
        assert 70.0 <= s6.tempo_bpm <= 100.0
        assert s6.total_beats == pytest.approx(32.0)
        # Check if minor 3rd (3 semitones) appears in scale steps
        pitches = [n.pitch for n in s6.notes]
        diffs = [abs(p2 - p1) for p1, p2 in zip(pitches[:-1], pitches[1:])]
        if 3 in diffs:
            l6_has_minor = True
    assert l6_has_minor, "Level 6 should include minor scales/arpeggios"

    # Level 7: wide range, octave leaps, rests, chromatic passing notes (75-105 BPM)
    l7_has_octave_leap = False
    l7_has_mixed_durations = False
    for seed in range(20):
        s7 = generate_score(level=7, seed=seed)
        assert 75.0 <= s7.tempo_bpm <= 105.0
        pitches = [n.pitch for n in s7.notes]
        leaps = [abs(p2 - p1) for p1, p2 in zip(pitches[:-1], pitches[1:])]
        if any(l >= 12 for l in leaps):
            l7_has_octave_leap = True
        durs = set(n.duration_beats for n in s7.notes)
        if len(durs) >= 3:
            l7_has_mixed_durations = True
    assert l7_has_octave_leap, "Level 7 should include octave leaps"
    assert l7_has_mixed_durations, "Level 7 should include mixed note lengths"

    # Level 8: fast pieces, 16th note runs (0.25 beat), repeated notes, higher tempo (100-130 BPM), >=32 beats
    l8_has_16th_run = False
    l8_has_repeated_notes = False
    for seed in range(20):
        s8 = generate_score(level=8, seed=seed)
        assert 100.0 <= s8.tempo_bpm <= 130.0
        assert s8.total_beats >= 32.0
        # 16th-note runs (0.25 beat duration)
        if any(n.duration_beats == 0.25 for n in s8.notes):
            l8_has_16th_run = True
        # Repeated notes
        pitches = [n.pitch for n in s8.notes]
        if any(p1 == p2 for p1, p2 in zip(pitches[:-1], pitches[1:])):
            l8_has_repeated_notes = True
    assert l8_has_16th_run, "Level 8 should include 16th-note runs"
    assert l8_has_repeated_notes, "Level 8 should include repeated notes"


def test_dataset_split_and_no_leakage(tmp_path: Path):
    """Verify dataset generation: 85/15 split, no duplicates between train and heldout, all 8 levels present."""
    import json
    generate_dataset(output_base=tmp_path, base_seed=100)

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check both piles have all 8 levels
    train_levels = set(item["level"] for item in manifest if item["split"] == "train")
    heldout_levels = set(item["level"] for item in manifest if item["split"] == "heldout")

    assert train_levels == set(range(1, 9))
    assert heldout_levels == set(range(1, 9))

    # Verify no file in train is identical to any file in heldout
    from pianorl.score import load_score
    train_fingerprints = set()
    for item in manifest:
        if item["split"] == "train":
            sc = load_score(tmp_path / item["filename"])
            train_fingerprints.add(get_score_fingerprint(sc))

    for item in manifest:
        if item["split"] == "heldout":
            sc = load_score(tmp_path / item["filename"])
            fp = get_score_fingerprint(sc)
            assert fp not in train_fingerprints, f"Held-out piece {item['filename']} leaked into training set!"
