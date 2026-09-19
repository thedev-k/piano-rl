import numpy as np
import pytest
from pathlib import Path

from pianorl.score import (
    Score,
    NoteEvent,
    load_score,
    ScoreWindow,
    MIN_PIANO_PITCH,
    MAX_PIANO_PITCH,
    NUM_PIANO_KEYS,
)


@pytest.fixture(scope="module")
def c_major_scale_score() -> Score:
    midi_path = Path("data/examples/c_major_scale.mid")
    return load_score(midi_path)


def test_window_shape(c_major_scale_score: Score):
    window_engine = ScoreWindow(c_major_scale_score, n_beats=4, steps_per_beat=4)

    # Standard check at step 0
    w0 = window_engine.get_window(current_step=0)
    assert w0.shape == (2, 88, 16)
    assert w0.dtype == np.float32

    # Check at intermediate step
    w4 = window_engine.get_window(current_step=4)
    assert w4.shape == (2, 88, 16)

    # Check past the end of the song
    w100 = window_engine.get_window(current_step=100)
    assert w100.shape == (2, 88, 16)


def test_c4_row_mapping(c_major_scale_score: Score):
    """Verify key row math: row = pitch - 21. C4 (pitch 60) must land on row 39."""
    window_engine = ScoreWindow(c_major_scale_score, n_beats=4, steps_per_beat=4)
    window = window_engine.get_window(current_step=0)

    # C4 is pitch 60: 60 - 21 = 39
    c4_row = 60 - MIN_PIANO_PITCH
    assert c4_row == 39
    assert window[0, c4_row, 0] == 1.0  # C4 onset at slot 0

    # Bounds check
    assert (MIN_PIANO_PITCH - 21) == 0      # A0 -> row 0
    assert (MAX_PIANO_PITCH - 21) == 87     # C8 -> row 87


def test_c_major_scale_step_0(c_major_scale_score: Score):
    """At step 0: 4 notes fall in the 16-slot window starting at slots 0, 4, 8, and 12."""
    window_engine = ScoreWindow(c_major_scale_score, n_beats=4, steps_per_beat=4)
    window = window_engine.get_window(current_step=0)

    expected_notes = [
        (60, 0),   # C4 at slot 0
        (62, 4),   # D4 at slot 4
        (64, 8),   # E4 at slot 8
        (65, 12),  # F4 at slot 12
    ]

    for pitch, slot in expected_notes:
        row = pitch - 21
        # Channel 0: Onset must be 1.0 only at the start slot
        assert window[0, row, slot] == 1.0
        # No onset on other slots for this note
        for other_slot in range(16):
            if other_slot != slot:
                assert window[0, row, other_slot] == 0.0

        # Channel 1: Note must be sounding for all 4 slots (1 beat)
        for sound_slot in range(slot, slot + 4):
            assert window[1, row, sound_slot] == 1.0


def test_note_at_edge_just_past_window_not_visible(c_major_scale_score: Score):
    """G4 (pitch 67) starts at beat 4 (step 16), exactly one slot past the 16-slot window [0..15].

    It must NOT appear anywhere in the window.
    """
    window_engine = ScoreWindow(c_major_scale_score, n_beats=4, steps_per_beat=4)
    window = window_engine.get_window(current_step=0)

    g4_row = 67 - 21
    # G4 must not have any onsets or sounding slots in window 0
    assert window[0, g4_row, :].sum() == 0.0
    assert window[1, g4_row, :].sum() == 0.0


def test_long_note_started_before_window():
    """A note that started before current_step must show in channel 1 (held) but NOT channel 0 (onset)."""
    # Create a 3-beat note (12 steps) from beat 0 to beat 3
    score = Score(notes=[NoteEvent(pitch=60, start_beat=0.0, duration_beats=3.0)])
    window_engine = ScoreWindow(score, n_beats=4, steps_per_beat=4)

    # Window starts at step 4 (beat 1.0)
    window = window_engine.get_window(current_step=4)
    c4_row = 60 - 21

    # Channel 0: No onset in this window (note started at step 0)
    assert window[0, c4_row, :].sum() == 0.0

    # Channel 1: Note was 12 steps long, started at 0, ends at step 12.
    # From step 4 to 11 (slots 0 to 7) it is still sounding: 8 slots.
    for slot in range(8):
        assert window[1, c4_row, slot] == 1.0
    for slot in range(8, 16):
        assert window[1, c4_row, slot] == 0.0


def test_window_past_end_of_piece_all_zeros(c_major_scale_score: Score):
    """Past the end of the song, the window must be completely zeros with exact same shape."""
    window_engine = ScoreWindow(c_major_scale_score, n_beats=4, steps_per_beat=4)

    # Song lasts 32 steps (8 beats * 4)
    window_past_end = window_engine.get_window(current_step=50)

    assert window_past_end.shape == (2, 88, 16)
    assert np.all(window_past_end == 0.0)


def test_out_of_range_pitches_ignored():
    """Notes with pitch < 21 or > 108 must be ignored with a warning and not crash."""
    score = Score(
        notes=[
            NoteEvent(pitch=15, start_beat=0.0, duration_beats=1.0),   # Below 21
            NoteEvent(pitch=120, start_beat=1.0, duration_beats=1.0),  # Above 108
            NoteEvent(pitch=60, start_beat=2.0, duration_beats=1.0),   # Valid C4
        ]
    )
    window_engine = ScoreWindow(score, n_beats=4, steps_per_beat=4)

    with pytest.warns(UserWarning, match="outside the 88-key piano range"):
        window = window_engine.get_window(current_step=0)

    # Only C4 should appear in the window (at slot 8)
    assert window[0, :, :].sum() == 1.0
    assert window[0, 60 - 21, 8] == 1.0


def test_changing_n_beats(c_major_scale_score: Score):
    """Setting n_beats to 2 changes slots to 8 (2 * 4)."""
    window_engine = ScoreWindow(c_major_scale_score, n_beats=2, steps_per_beat=4)

    assert window_engine.num_slots == 8
    window = window_engine.get_window(current_step=0)
    assert window.shape == (2, 88, 8)


def test_total_steps(c_major_scale_score: Score):
    """8 notes of 1 beat each = 8 beats * 4 steps/beat = 32 total steps."""
    window_engine = ScoreWindow(c_major_scale_score, n_beats=4, steps_per_beat=4)
    assert window_engine.total_steps() == 32
