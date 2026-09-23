from pathlib import Path
from typing import List, Tuple
import mido


def create_midi_file(
    output_path: Path,
    notes: List[Tuple[int, float, float]],
    tempo_bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> None:
    """Create and save a standard MIDI file from a list of note tuples.

    Args:
        output_path: Target path to save the .mid file.
        notes: List of tuples (pitch, start_beat, duration_beats).
        tempo_bpm: Tempo in beats per minute.
        ticks_per_beat: MIDI resolution (ticks per beat).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set initial tempo
    track.append(
        mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0)
    )

    # Convert notes to absolute tick events
    events: List[Tuple[int, str, int, int]] = []
    for item in notes:
        pitch = item[0]
        start_beat = item[1]
        duration_beats = item[2]
        vel = item[3] if len(item) > 3 else 64
        start_tick = int(round(start_beat * ticks_per_beat))
        end_tick = int(round((start_beat + duration_beats) * ticks_per_beat))
        events.append((start_tick, "note_on", pitch, vel))
        events.append((end_tick, "note_off", pitch, 0))

    # Sort events: primary by tick, secondary note_off before note_on at same tick
    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    # Add messages with delta times
    last_tick = 0
    for tick, event_type, pitch, velocity in events:
        delta_ticks = tick - last_tick
        track.append(
            mido.Message(
                event_type, note=pitch, velocity=velocity, time=delta_ticks
            )
        )
        last_tick = tick

    mid.save(str(output_path))
    print(f"Created MIDI file: {output_path} ({len(notes)} notes, {tempo_bpm} BPM)")


def main():
    examples_dir = Path("data/examples")

    # 1. C major scale (8 notes, one beat each, beats 0 through 7)
    # C4, D4, E4, F4, G4, A4, B4, C5
    scale_pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    scale_notes = [
        (pitch, float(beat), 1.0)
        for beat, pitch in enumerate(scale_pitches)
    ]
    create_midi_file(
        examples_dir / "c_major_scale.mid",
        notes=scale_notes,
        tempo_bpm=120.0,
    )

    # 2. First 2 bars of "Twinkle, Twinkle, Little Star" in C major (4/4 time)
    # Bar 1: C4, C4, G4, G4 (1 beat each)
    # Bar 2: A4, A4 (1 beat each), G4 (2 beats - half note)
    melody_notes = [
        (60, 0.0, 1.0),  # C4
        (60, 1.0, 1.0),  # C4
        (67, 2.0, 1.0),  # G4
        (67, 3.0, 1.0),  # G4
        (69, 4.0, 1.0),  # A4
        (69, 5.0, 1.0),  # A4
        (67, 6.0, 2.0),  # G4 (half note, 2 beats)
    ]
    create_midi_file(
        examples_dir / "melody.mid",
        notes=melody_notes,
        tempo_bpm=100.0,
    )


if __name__ == "__main__":
    main()
