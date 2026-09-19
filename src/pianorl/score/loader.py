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
    first_tempo_bpm: Optional[float] = None
    tempo_warning_issued = False

    # Process all tracks in the MIDI file
    for track in mid.tracks:
        abs_tick = 0
        active_notes: Dict[Tuple[int, int], List[int]] = {}

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

            # Handle note onset
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active_notes.setdefault(key, []).append(abs_tick)

            # Handle note release (note_off or note_on with velocity 0)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active_notes and active_notes[key]:
                    start_tick = active_notes[key].pop(0)
                    duration_ticks = abs_tick - start_tick
                    if duration_ticks > 0:
                        start_beat = start_tick / ticks_per_beat
                        duration_beats = duration_ticks / ticks_per_beat
                        notes.append(
                            NoteEvent(
                                pitch=msg.note,
                                start_beat=round(start_beat, 6),
                                duration_beats=round(duration_beats, 6),
                            )
                        )

        # Close any lingering notes that didn't receive an explicit note_off
        for (channel, pitch), starts in active_notes.items():
            for start_tick in starts:
                duration_ticks = abs_tick - start_tick
                if duration_ticks > 0:
                    start_beat = start_tick / ticks_per_beat
                    duration_beats = duration_ticks / ticks_per_beat
                    notes.append(
                        NoteEvent(
                            pitch=pitch,
                            start_beat=round(start_beat, 6),
                            duration_beats=round(duration_beats, 6),
                        )
                    )

    # Sort note events primarily by start_beat, secondarily by pitch
    notes.sort(key=lambda n: (n.start_beat, n.pitch))

    tempo = first_tempo_bpm if first_tempo_bpm is not None else 120.0
    return Score(notes=notes, tempo_bpm=tempo)
