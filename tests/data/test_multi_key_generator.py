"""Unit tests for multi-key procedural music generator (Levels 1M to 8M)."""

import math
import pytest

from pianorl.score import (
    Score,
    NoteEvent,
    MIN_PIANO_PITCH,
    MAX_PIANO_PITCH,
)
from pianorl.data.multi_key_generator import (
    generate_multi_key_score,
    generate_level_1m,
    generate_level_2m,
    generate_level_3m,
    generate_level_4m,
    generate_level_5m,
    generate_level_6m,
    generate_level_7m,
    generate_level_8m,
)


def has_polyphony(score: Score) -> bool:
    """Check if a score has any simultaneous or overlapping notes."""
    # Check simultaneous onsets
    start_counts: dict[float, int] = {}
    for n in score.notes:
        b = round(n.start_beat, 4)
        start_counts[b] = start_counts.get(b, 0) + 1
        if start_counts[b] > 1:
            return True

    # Check temporal overlap
    for i, n1 in enumerate(score.notes):
        end1 = n1.start_beat + n1.duration_beats
        for n2 in score.notes[i + 1 :]:
            if n2.start_beat >= end1:
                break
            if n2.pitch != n1.pitch:
                return True
    return False


def test_seed_reproducibility():
    """Verify that same seed produces identical polyphonic scores."""
    for level in range(1, 9):
        score_a = generate_multi_key_score(level, seed=42)
        score_b = generate_multi_key_score(f"{level}M", seed=42)

        assert score_a.tempo_bpm == score_b.tempo_bpm
        assert len(score_a.notes) == len(score_b.notes)
        for na, nb in zip(score_a.notes, score_b.notes):
            assert na.pitch == nb.pitch
            assert na.start_beat == pytest.approx(nb.start_beat)
            assert na.duration_beats == pytest.approx(nb.duration_beats)


def test_level_name_formats():
    """Verify level parameter accepts integers and string variants ('1M', '1m', 1)."""
    s_int = generate_multi_key_score(3, seed=123)
    s_str_upper = generate_multi_key_score("3M", seed=123)
    s_str_lower = generate_multi_key_score("3m", seed=123)

    assert len(s_int.notes) == len(s_str_upper.notes) == len(s_str_lower.notes)
    assert s_int.notes[0].pitch == s_str_upper.notes[0].pitch == s_str_lower.notes[0].pitch

    with pytest.raises(ValueError):
        generate_multi_key_score(0)
    with pytest.raises(ValueError):
        generate_multi_key_score(10)
    with pytest.raises(ValueError):
        generate_multi_key_score("10M")
    with pytest.raises(ValueError):
        generate_multi_key_score("invalid")
    with pytest.raises(TypeError):
        generate_multi_key_score(3.5)  # type: ignore


def test_pitches_within_88_keys():
    """Verify all notes across all 9 multi-key levels stay strictly within piano keys [21, 108]."""
    for level in range(1, 10):
        for seed in range(5):
            score = generate_multi_key_score(level, seed=seed * 77)
            assert len(score.notes) > 0
            for note in score.notes:
                assert MIN_PIANO_PITCH <= note.pitch <= MAX_PIANO_PITCH, (
                    f"Level {level} note pitch {note.pitch} out of bounds"
                )


def test_beats_on_16th_grid():
    """Verify note start beats and durations align strictly to 0.25-beat grid for Levels 1M-8M."""
    for level in range(1, 9):
        for seed in range(5):
            score = generate_multi_key_score(level, seed=seed)
            for note in score.notes:
                rem_start = round(note.start_beat % 0.25, 4)
                assert rem_start == 0.0 or math.isclose(rem_start, 0.25, abs_tol=1e-5), (
                    f"Level {level} start_beat {note.start_beat} not on 16th grid"
                )
                rem_dur = round(note.duration_beats % 0.25, 4)
                assert rem_dur == 0.0 or math.isclose(rem_dur, 0.25, abs_tol=1e-5), (
                    f"Level {level} duration_beats {note.duration_beats} not on 16th grid"
                )


def test_all_levels_have_polyphony():
    """Verify every level (1M to 9M) produces polyphonic scores with simultaneous/overlapping notes."""
    for level in range(1, 10):
        for seed in range(3):
            score = generate_multi_key_score(level, seed=seed)
            assert has_polyphony(score), f"Level {level}M failed to produce polyphony!"


def test_level_1m_dyads():
    """Level 1M should feature exactly 2 notes sounding simultaneously at each beat."""
    score = generate_multi_key_score(1, seed=10)
    assert len(score.notes) == 32  # 16 beats * 2 notes per beat
    # Every start beat should have exactly 2 notes
    from collections import Counter
    onsets = Counter(round(n.start_beat, 2) for n in score.notes)
    for beat, count in onsets.items():
        assert count == 2, f"Expected 2 notes at beat {beat}, found {count}"


def test_level_2m_bass_pedal_and_moving_treble():
    """Level 2M should feature sustained bass notes (durations >= 2.0 beats) under moving treble."""
    score = generate_multi_key_score(2, seed=20)
    durations = [n.duration_beats for n in score.notes]
    assert any(d >= 2.0 for d in durations), "Level 2M must feature sustained bass pedal notes"


def test_level_3m_triads():
    """Level 3M should feature 3-note triads in the left hand (3 simultaneous bass chord notes)."""
    score = generate_multi_key_score(3, seed=30)
    from collections import Counter
    onsets = Counter(round(n.start_beat, 2) for n in score.notes)
    # At chord onset beats, at least 3 notes start simultaneously
    assert any(count >= 3 for count in onsets.values()), "Level 3M must contain simultaneous 3-note chords"


def test_level_8m_fast_polyphony_runs_and_chords():
    """Level 8M should feature fast tempo (100-130 BPM), 16th-note runs (0.25 beat), and chord punches."""
    score = generate_multi_key_score(8, seed=80)
    assert 100.0 <= score.tempo_bpm <= 130.0
    assert any(n.duration_beats == 0.25 for n in score.notes), "Level 8M must have 16th-note runs"


def test_level_9m_dense_sustained_chords():
    """Level 9M: dense sustained bed + fast moving line targeting ~6-7 chord moments per beat."""
    from collections import Counter
    from pianorl.score.melody import analyze_score

    all_pitches = set()
    densities = []

    for seed in range(10):
        score = generate_multi_key_score(9, seed=seed * 31 + 7)
        assert 75.0 <= score.tempo_bpm <= 105.0
        assert len(score.notes) > 100

        analysis = analyze_score(score)
        chord_density = analysis["num_chords"] / score.total_beats
        densities.append(chord_density)

        # Confirm presence of sustained bed (>= 2.0 beats)
        has_sustained = any(n.duration_beats >= 2.0 for n in score.notes)
        assert has_sustained, f"Seed {seed} missing sustained chord bed!"

        # Confirm presence of fast moving notes (<= 0.25 down to 0.125 beats)
        has_fast_moving = any(n.duration_beats <= 0.25 for n in score.notes)
        assert has_fast_moving, f"Seed {seed} missing fast moving melodic line!"

        for n in score.notes:
            all_pitches.add(n.pitch)

    # Verify chord density is roughly 6 to 7 chord moments per beat
    avg_density = sum(densities) / len(densities)
    assert 5.5 <= avg_density <= 7.5, (
        f"Level 9M expected ~6-7 chord moments/beat, got {avg_density:.2f}"
    )

    # Verify full keyboard range coverage across seeds (bass < 36 up to treble > 88)
    assert min(all_pitches) < 36, f"Lowest pitch {min(all_pitches)} not in low bass range"
    assert max(all_pitches) > 88, f"Highest pitch {max(all_pitches)} not in high treble range"


def test_multikey_manifest_split_sizes_include_9m():
    """Verify data/manifest_multikey.json includes 170 train and 30 held-out pieces for 9M."""
    import json
    from pathlib import Path

    manifest_path = Path("data/manifest_multikey.json")
    if not manifest_path.exists():
        pytest.skip("Manifest not generated yet.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items_9m_train = [it for it in manifest if it["level"] == "9M" and it["split"] == "train"]
    items_9m_heldout = [it for it in manifest if it["level"] == "9M" and it["split"] == "heldout"]

    assert len(items_9m_train) == 170, f"Expected 170 train pieces for 9M, got {len(items_9m_train)}"
    assert len(items_9m_heldout) == 30, f"Expected 30 heldout pieces for 9M, got {len(items_9m_heldout)}"

