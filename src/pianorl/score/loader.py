import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import mido

from .score import Score, NoteEvent


def load_score(file_path: str | Path) -> Score:
    """Load a MIDI (.mid) file and return a Score object.

    Notes are converted directly from MIDI ticks to beats using the file's
    ticks_per_beat resolution, keeping beat timing exact.

    Args:
        file_path: Path to the .mid file.

    Returns:
        Score: A Score object containing sorted NoteEvents and tempo in BPM.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, corrupted, or has invalid data.
    """
    path = Path(file_path)

    # Check for missing file
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: '{path}'")

    # Check for empty file (0 bytes)
    if path.stat().st_size == 0:
        raise ValueError(f"MIDI file is empty (0 bytes): '{path}'")

    # Parse with mido
    try:
        mid = mido.MidiFile(str(path))
    except Exception as e:
        raise ValueError(f"Failed to parse MIDI file '{path}': {e}") from e

    ticks_per_beat = mid.ticks_per_beat
    if ticks_per_beat <= 0:
        raise ValueError(f"Invalid ticks_per_beat in MIDI file: {ticks_per_beat}")

    notes: List[NoteEvent] = []
    pedal_intervals: List[Tuple[float, float]] = []
    first_tempo_bpm: Optional[float] = None
    tempo_warning_issued = False

    # Process all tracks in the MIDI file
    for track in mid.tracks:
        abs_tick = 0
        active_notes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        pedal_down_tick: Optional[int] = None

        for msg in track:
            abs_tick += msg.time

            # Handle tempo meta messages
            if msg.is_meta and msg.type == "set_tempo":
                current_bpm = round(mido.tempo2bpm(msg.tempo), 2)
                if first_tempo_bpm is None:
                    first_tempo_bpm = current_bpm
                elif abs(current_bpm - first_tempo_bpm) > 0.01 and not tempo_warning_issued:
                    warning_msg = (
                        f"Tempo change detected in '{path.name}' ({current_bpm} BPM). "
                        f"Keeping initial tempo of {first_tempo_bpm} BPM."
                    )
                    print(f"WARNING: {warning_msg}")
                    warnings.warn(warning_msg, UserWarning, stacklevel=2)
                    tempo_warning_issued = True

            # Handle sustain pedal (Control Change 64)
            elif msg.type == "control_change" and msg.control == 64:
                if msg.value >= 64:
                    if pedal_down_tick is None:
                        pedal_down_tick = abs_tick
                else:
                    if pedal_down_tick is not None:
                        p_start = round(pedal_down_tick / ticks_per_beat, 6)
                        p_end = round(abs_tick / ticks_per_beat, 6)
                        if p_end > p_start:
                            pedal_intervals.append((p_start, p_end))
                        pedal_down_tick = None

            # Handle note onset
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active_notes.setdefault(key, []).append((abs_tick, msg.velocity))

            # Handle note release (note_off or note_on with velocity 0)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes and active_notes[key]:
                    start_tick, velocity = active_notes[key].pop(0)
                    duration_ticks = abs_tick - start_tick
                    if duration_ticks > 0:
                        start_beat = start_tick / ticks_per_beat
                        duration_beats = duration_ticks / ticks_per_beat
                        notes.append(
                            NoteEvent(
                                pitch=msg.note,
                                start_beat=round(start_beat, 6),
                                duration_beats=round(duration_beats, 6),
                                velocity=velocity,
                            )
                        )

        # Close any active pedal still down at track end
        if pedal_down_tick is not None:
            p_start = round(pedal_down_tick / ticks_per_beat, 6)
            p_end = round(abs_tick / ticks_per_beat, 6)
            if p_end > p_start:
                pedal_intervals.append((p_start, p_end))
            pedal_down_tick = None

        # Close any lingering notes that didn't receive an explicit note_off
        for (channel, pitch), starts in active_notes.items():
            for start_tick, velocity in starts:
                duration_ticks = abs_tick - start_tick
                if duration_ticks > 0:
                    start_beat = start_tick / ticks_per_beat
                    duration_beats = duration_ticks / ticks_per_beat
                    notes.append(
                        NoteEvent(
                            pitch=pitch,
                            start_beat=round(start_beat, 6),
                            duration_beats=round(duration_beats, 6),
                            velocity=velocity,
                        )
                    )

    # Sort note events primarily by start_beat, secondarily by pitch
    notes.sort(key=lambda n: (n.start_beat, n.pitch))
    pedal_intervals.sort(key=lambda p: (p[0], p[1]))

    tempo = first_tempo_bpm if first_tempo_bpm is not None else 120.0
    return Score(notes=notes, tempo_bpm=tempo, pedal_intervals=pedal_intervals)


def save_score_to_midi(
    score: Score,
    file_path: str | Path,
    ticks_per_beat: int = 480,
) -> None:
    """Save a Score object as a standard MIDI (.mid) file.

    Args:
        score: The Score object to write.
        file_path: Destination file path.
        ticks_per_beat: Ticks per beat resolution (default: 480).
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set tempo
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.tempo_bpm), time=0)
    )

    # Convert notes and pedal intervals to absolute tick events
    events: List[Tuple[int, str, int, int]] = []
    for note in score.notes:
        start_tick = int(round(note.start_beat * ticks_per_beat))
        end_tick = int(round(note.end_beat * ticks_per_beat))
        events.append((start_tick, "note_on", note.pitch, note.velocity))
        events.append((end_tick, "note_off", note.pitch, 0))

    for p_start, p_end in score.pedal_intervals:
        p_start_tick = int(round(p_start * ticks_per_beat))
        p_end_tick = int(round(p_end * ticks_per_beat))
        events.append((p_start_tick, "cc_64", 64, 127))
        events.append((p_end_tick, "cc_64", 64, 0))

    # Sort events: primary by tick, secondary note_off before note_on at same tick
    events.sort(key=lambda e: (e[0], 0 if e[1] in ("note_off", "cc_64") and e[3] == 0 else 1))

    # Add messages with delta times
    last_tick = 0
    for tick, event_type, pitch, velocity in events:
        delta_ticks = tick - last_tick
        if event_type == "cc_64":
            track.append(
                mido.Message("control_change", control=64, value=velocity, time=delta_ticks)
            )
        else:
            track.append(
                mido.Message(event_type, note=pitch, velocity=velocity, time=delta_ticks)
            )
        last_tick = tick

    mid.save(str(path))

