"""Tests for score analysis and melody extraction."""

import pytest
from pathlib import Path

from pianorl.score import (
    NoteEvent,
    Score,
    analyze_score,
    extract_melody,
    save_score_to_midi,
    load_score,
)


def test_analyze_score_empty():
    """Verify analyze_score handles empty scores gracefully."""
    score = Score(notes=[], tempo_bpm=120.0)
    info = analyze_score(score)
    assert info["num_notes"] == 0
    assert info["length_beats"] == 0.0
    assert info["num_chords"] == 0
    assert info["rounded_notes"] == 0
    assert info["lowest_key"] == "N/A"
    assert info["highest_key"] == "N/A"


def test_analyze_score_metrics():
    """Verify info line metrics: notes, range, chords, triplets/rounding."""
    notes = [
        # Chord at beat 0: C4 (60) and E4 (64)
        NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
        NoteEvent(pitch=64, start_beat=0.0, duration_beats=1.0),
        # Single note at beat 1.0: G4 (67)
        NoteEvent(pitch=67, start_beat=1.0, duration_beats=0.5),
        # Triplet note at beat 2.333: C5 (72) (not on 16th-note grid)
        NoteEvent(pitch=72, start_beat=2.333, duration_beats=0.333),
    ]
    score = Score(notes=notes, tempo_bpm=100.0)
    info = analyze_score(score)

    assert info["num_notes"] == 4
    assert info["tempo_bpm"] == 100.0
    assert info["lowest_key"] == "C4 (60)"
    assert info["highest_key"] == "C5 (72)"
    assert info["shortest_note_beats"] == pytest.approx(0.333, rel=1e-2)
    # Chord at beat 0
    assert info["num_chords"] == 1
    # Triplet note is rounded
    assert info["rounded_notes"] == 1


def test_extract_melody_chords():
    """Verify extract_melody keeps only the highest pitch note at simultaneous starts."""
    notes = [
        # Chord at beat 0: C4 (60), E4 (64), G4 (67) -> G4 should be kept
        NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
        NoteEvent(pitch=64, start_beat=0.0, duration_beats=1.0),
        NoteEvent(pitch=67, start_beat=0.0, duration_beats=1.0),
        # Single note at beat 1: D4 (62) -> D4 should be kept
        NoteEvent(pitch=62, start_beat=1.0, duration_beats=1.0),
        # Chord at beat 2: C4 (60), A4 (69) -> A4 should be kept
        NoteEvent(pitch=60, start_beat=2.0, duration_beats=1.0),
        NoteEvent(pitch=69, start_beat=2.0, duration_beats=1.0),
    ]
    score = Score(notes=notes, tempo_bpm=120.0)

    melody_score, dropped_count = extract_melody(score)

    # 3 chord notes were dropped (C4, E4 at beat 0; C4 at beat 2)
    assert dropped_count == 3
    assert len(melody_score.notes) == 3

    pitches = [n.pitch for n in melody_score.notes]
    assert pitches == [67, 62, 69]

    # Verify start beats
    beats = [n.start_beat for n in melody_score.notes]
    assert beats == [0.0, 1.0, 2.0]


def test_extract_melody_sustained_accompaniment():
    """Verify lower accompaniment notes underneath a sustained high melody note are dropped."""
    notes = [
        # Melody: High note held for 4 beats: C5 (72)
        NoteEvent(pitch=72, start_beat=0.0, duration_beats=4.0),
        # Lower accompaniment notes starting during the held melody note
        NoteEvent(pitch=48, start_beat=0.0, duration_beats=1.0),  # C3 chord onset
        NoteEvent(pitch=55, start_beat=1.0, duration_beats=1.0),  # G3 inner note
        NoteEvent(pitch=52, start_beat=2.0, duration_beats=1.0),  # E3 inner note
    ]
    score = Score(notes=notes, tempo_bpm=100.0)

    melody_score, dropped_count = extract_melody(score)

    assert dropped_count == 3
    assert len(melody_score.notes) == 1
    assert melody_score.notes[0].pitch == 72


def test_extract_melody_descending_legato():
    """Verify descending melody with slight legato overlap is not accidentally dropped."""
    notes = [
        # C5 held slightly past 1 beat (e.g. 1.05 beats)
        NoteEvent(pitch=72, start_beat=0.0, duration_beats=1.05),
        # B4 starting at beat 1.0
        NoteEvent(pitch=71, start_beat=1.0, duration_beats=1.0),
        # A4 starting at beat 2.0
        NoteEvent(pitch=69, start_beat=2.0, duration_beats=1.0),
    ]
    score = Score(notes=notes, tempo_bpm=120.0)

    melody_score, dropped_count = extract_melody(score)

    assert dropped_count == 0
    assert len(melody_score.notes) == 3
    assert [n.pitch for n in melody_score.notes] == [72, 71, 69]
