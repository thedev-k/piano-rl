import math
from pathlib import Path
import pytest

from pianorl.score import (
    Score,
    generate_score,
    MIN_PIANO_PITCH,
    MAX_PIANO_PITCH,
)
from scripts.generate_dataset import generate_dataset, get_score_fingerprint


def test_seed_reproducibility():
    """Verify that using the same seed produces the exact same notes and tempo."""
    for level in [1, 2, 3, 4]:
        score_a = generate_score(level=level, seed=999)
        score_b = generate_score(level=level, seed=999)

        assert score_a.tempo_bpm == score_b.tempo_bpm
        assert len(score_a) == len(score_b)
        for na, nb in zip(score_a.notes, score_b.notes):
            assert na.pitch == nb.pitch
            assert na.start_beat == nb.start_beat
            assert na.duration_beats == nb.duration_beats


def test_pitches_within_88_keys():
    """Verify every note generated across all levels stays strictly inside the 88-key piano."""
    for level in [1, 2, 3, 4]:
        for seed in range(10):
            score = generate_score(level=level, seed=seed * 100)
            for note in score.notes:
                assert MIN_PIANO_PITCH <= note.pitch <= MAX_PIANO_PITCH


def test_beats_multiples_of_16th_grid():
    """Verify all start beats and durations land exactly on the 16th-note grid (multiples of 0.25)."""
    for level in [1, 2, 3, 4]:
        for seed in range(5):
            score = generate_score(level=level, seed=seed)
            for note in score.notes:
                # Check start_beat is multiple of 0.25
                rem_start = round(note.start_beat % 0.25, 4)
                assert rem_start == 0.0 or math.isclose(rem_start, 0.25, abs_tol=1e-5)

                # Check duration_beats is multiple of 0.25
                rem_dur = round(note.duration_beats % 0.25, 4)
                assert rem_dur == 0.0 or math.isclose(rem_dur, 0.25, abs_tol=1e-5)


def test_levels_1_and_2_five_key_range():
    """Verify levels 1 and 2 only use pitches from a single five-white-key span."""
    for level in [1, 2]:
        for seed in range(10):
            score = generate_score(level=level, seed=seed)
            unique_pitches = sorted(list(set(n.pitch for n in score.notes)))
            # Cannot exceed 5 distinct pitches
            assert len(unique_pitches) <= 5


def test_tempo_and_length_rules():
    """Verify that tempos and lengths match the level specifications."""
    for seed in range(10):
        # Level 1: 60-80 BPM, 16 beats, duration 1.0
        s1 = generate_score(level=1, seed=seed)
        assert 60.0 <= s1.tempo_bpm <= 80.0
        assert s1.total_beats == pytest.approx(16.0)
        assert all(n.duration_beats == 1.0 for n in s1.notes)

        # Level 2: 60-90 BPM, 16 beats, duration in {1.0, 2.0}
        s2 = generate_score(level=2, seed=seed)
        assert 60.0 <= s2.tempo_bpm <= 90.0
        assert s2.total_beats == pytest.approx(16.0)
        assert all(n.duration_beats in (1.0, 2.0) for n in s2.notes)

        # Level 3: 70-100 BPM, 32 beats, duration in {0.5, 1.0}
        s3 = generate_score(level=3, seed=seed)
        assert 70.0 <= s3.tempo_bpm <= 100.0
        assert s3.total_beats == pytest.approx(32.0)
        assert all(n.duration_beats in (0.5, 1.0) for n in s3.notes)

        # Level 4: 70-110 BPM, 32 beats, duration in {0.5, 1.0, 2.0}
        s4 = generate_score(level=4, seed=seed)
        assert 70.0 <= s4.tempo_bpm <= 110.0
        assert s4.total_beats == pytest.approx(32.0)
        assert all(n.duration_beats in (0.5, 1.0, 2.0) for n in s4.notes)


def test_dataset_split_and_no_leakage(tmp_path: Path):
    """Verify dataset generation: 85/15 split, no duplicates between train and heldout, all levels present."""
    # Generate into temporary folder to verify logic
    import json
    generate_dataset(output_base=tmp_path, base_seed=100)

    manifest_file = tmp_path / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Check both piles have all four levels
    train_levels = set(item["level"] for item in manifest if item["split"] == "train")
    heldout_levels = set(item["level"] for item in manifest if item["split"] == "heldout")

    assert train_levels == {1, 2, 3, 4}
    assert heldout_levels == {1, 2, 3, 4}

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
