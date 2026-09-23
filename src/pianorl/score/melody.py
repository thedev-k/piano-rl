"""Melody extraction and musical score analysis utilities."""

from collections import Counter
from typing import Dict, List, Tuple

from .score import NoteEvent, Score, pitch_to_note_name


def analyze_score(score: Score, steps_per_beat: int = 4) -> dict:
    """Analyze a Score and return descriptive musical properties.

    Args:
        score: The Score object to analyze.
        steps_per_beat: Environment time-step resolution (default 4 for 16th-notes).

    Returns:
        dict containing:
            - num_notes: Total note count
            - length_beats: Score duration in beats
            - tempo_bpm: Playback tempo in BPM
            - lowest_key: Name and pitch number of the lowest note (e.g. 'C4 (60)')
            - lowest_pitch: Lowest MIDI pitch number
            - highest_key: Name and pitch number of the highest note (e.g. 'G5 (79)')
            - highest_pitch: Highest MIDI pitch number
            - shortest_note_beats: Duration of shortest note in beats
            - num_chords: Count of moments where multiple notes start at the same time
            - rounded_notes: Count of notes whose start beat does not align to 16th-notes
    """
    if not score.notes:
        return {
            "num_notes": 0,
            "length_beats": 0.0,
            "tempo_bpm": round(score.tempo_bpm, 1),
            "lowest_key": "N/A",
            "lowest_pitch": None,
            "highest_key": "N/A",
            "highest_pitch": None,
            "shortest_note_beats": 0.0,
            "num_chords": 0,
            "rounded_notes": 0,
        }

    pitches = [n.pitch for n in score.notes]
    min_p = min(pitches)
    max_p = max(pitches)
    shortest_dur = min(n.duration_beats for n in score.notes)

    # Chords: moments where more than 1 note starts simultaneously
    start_counts = Counter(round(n.start_beat, 4) for n in score.notes)
    num_chords = sum(1 for count in start_counts.values() if count > 1)

    # Notes rounded to 16th-note grid
    rounded_notes = sum(
        1
        for n in score.notes
        if abs(n.start_beat * steps_per_beat - round(n.start_beat * steps_per_beat)) > 1e-3
    )

    return {
        "num_notes": len(score.notes),
        "length_beats": round(score.total_beats, 2),
        "tempo_bpm": round(score.tempo_bpm, 1),
        "lowest_key": f"{pitch_to_note_name(min_p)} ({min_p})",
        "lowest_pitch": min_p,
        "highest_key": f"{pitch_to_note_name(max_p)} ({max_p})",
        "highest_pitch": max_p,
        "shortest_note_beats": round(shortest_dur, 3),
        "num_chords": num_chords,
        "rounded_notes": rounded_notes,
    }


def extract_melody(score: Score, steps_per_beat: int = 4) -> Tuple[Score, int]:
    """Extract a single-line melody from a score by keeping only the highest note at each moment.

    Drops harmony notes in chords, lower accompaniment tracks, and inner voices,
    producing a monophonic score suitable for single-key piano playback.

    Args:
        score: The original Score object.
        steps_per_beat: Environment time-step resolution (default 4).

    Returns:
        Tuple of:
            - melody_score: Monophonic Score with lower notes removed
            - dropped_notes: Total number of notes removed
    """
    if not score.notes:
        return Score(notes=[], tempo_bpm=score.tempo_bpm), 0

    # 1. Group notes by discrete start step
    step_to_notes: Dict[int, List[NoteEvent]] = {}
    for n in score.notes:
        step = int(round(n.start_beat * steps_per_beat))
        step_to_notes.setdefault(step, []).append(n)

    # For every start step, select candidate with the highest pitch
    candidates: List[NoteEvent] = []
    for step in sorted(step_to_notes.keys()):
        notes_at_step = step_to_notes[step]
        # Sort descending by pitch, then descending by duration
        notes_at_step.sort(key=lambda n: (n.pitch, n.duration_beats), reverse=True)
        candidates.append(notes_at_step[0])

    # 2. Filter out notes sounding underneath higher sustained notes,
    # and truncate overlapping notes for a clean monophonic melody line.
    final_notes: List[NoteEvent] = []

    for cand in candidates:
        cand_start_step = int(round(cand.start_beat * steps_per_beat))

        if not final_notes:
            final_notes.append(cand)
            continue

        prev = final_notes[-1]
        prev_start_step = int(round(prev.start_beat * steps_per_beat))
        prev_dur_steps = max(1, int(round(prev.duration_beats * steps_per_beat)))
        prev_end_step = prev_start_step + prev_dur_steps

        # Check if candidate starts while previous note is still sounding
        if cand_start_step < prev_end_step:
            if cand.pitch <= prev.pitch:
                # If it's a deep accompaniment note or significantly held chord:
                # (more than 7 semitones below, or sustained chord > 1 step overlap)
                overlap_steps = prev_end_step - cand_start_step
                semitone_diff = prev.pitch - cand.pitch
                if semitone_diff > 7 or overlap_steps > 1:
                    # Drop candidate as accompaniment
                    continue
                else:
                    # Legato melody transition: shorten previous note to candidate start
                    new_dur = max(0.25 / steps_per_beat, cand.start_beat - prev.start_beat)
                    final_notes[-1] = NoteEvent(
                        pitch=prev.pitch,
                        start_beat=prev.start_beat,
                        duration_beats=new_dur,
                        velocity=prev.velocity,
                    )
                    final_notes.append(cand)
            else:
                # Candidate is higher pitch: takes over melody
                new_dur = max(0.25 / steps_per_beat, cand.start_beat - prev.start_beat)
                final_notes[-1] = NoteEvent(
                    pitch=prev.pitch,
                    start_beat=prev.start_beat,
                    duration_beats=new_dur,
                    velocity=prev.velocity,
                )
                final_notes.append(cand)
        else:
            final_notes.append(cand)

    dropped_notes = len(score.notes) - len(final_notes)
    melody_score = Score(
        notes=final_notes,
        tempo_bpm=score.tempo_bpm,
        pedal_intervals=score.pedal_intervals,
    )
    return melody_score, dropped_notes
