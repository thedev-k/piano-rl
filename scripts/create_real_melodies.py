"""Generate 4 public-domain real piano melodies as MIDI files in data/real/ using pretty_midi."""

from pathlib import Path
from typing import List, Tuple
import pretty_midi


def create_midi_file(
    output_path: Path,
    tempo_bpm: float,
    notes: List[Tuple[int, float, float]],  # (pitch, start_beat, duration_beats)
) -> None:
    """Create and write a single-instrument MIDI file using pretty_midi."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
    piano_program = pretty_midi.instrument_name_to_program("Acoustic Grand Piano")
    piano = pretty_midi.Instrument(program=piano_program)

    seconds_per_beat = 60.0 / tempo_bpm

    for pitch, start_beat, duration_beats in notes:
        start_time = start_beat * seconds_per_beat
        end_time = (start_beat + duration_beats) * seconds_per_beat
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start_time,
            end=end_time,
        )
        piano.notes.append(note)

    pm.instruments.append(piano)
    pm.write(str(output_path))
    print(f"Created '{output_path}' ({len(notes)} notes, {tempo_bpm} BPM)")


def build_twinkle_twinkle() -> List[Tuple[int, float, float]]:
    """Twinkle, Twinkle, Little Star (16 beats, 14 notes)."""
    return [
        (60, 0.0, 1.0),   # C4
        (60, 1.0, 1.0),   # C4
        (67, 2.0, 1.0),   # G4
        (67, 3.0, 1.0),   # G4
        (69, 4.0, 1.0),   # A4
        (69, 5.0, 1.0),   # A4
        (67, 6.0, 2.0),   # G4
        (65, 8.0, 1.0),   # F4
        (65, 9.0, 1.0),   # F4
        (64, 10.0, 1.0),  # E4
        (64, 11.0, 1.0),  # E4
        (62, 12.0, 1.0),  # D4
        (62, 13.0, 1.0),  # D4
        (60, 14.0, 2.0),  # C4
    ]


def build_ode_to_joy() -> List[Tuple[int, float, float]]:
    """Beethoven's Ode to Joy (32 beats, 30 notes)."""
    return [
        # Phrase 1 (beats 0 - 16)
        (64, 0.0, 1.0),   # E4
        (64, 1.0, 1.0),   # E4
        (65, 2.0, 1.0),   # F4
        (67, 3.0, 1.0),   # G4
        (67, 4.0, 1.0),   # G4
        (65, 5.0, 1.0),   # F4
        (64, 6.0, 1.0),   # E4
        (62, 7.0, 1.0),   # D4
        (60, 8.0, 1.0),   # C4
        (60, 9.0, 1.0),   # C4
        (62, 10.0, 1.0),  # D4
        (64, 11.0, 1.0),  # E4
        (64, 12.0, 1.5),  # E4
        (62, 13.5, 0.5),  # D4
        (62, 14.0, 2.0),  # D4
        # Phrase 2 (beats 16 - 32)
        (64, 16.0, 1.0),  # E4
        (64, 17.0, 1.0),  # E4
        (65, 18.0, 1.0),  # F4
        (67, 19.0, 1.0),  # G4
        (67, 20.0, 1.0),  # G4
        (65, 21.0, 1.0),  # F4
        (64, 22.0, 1.0),  # E4
        (62, 23.0, 1.0),  # D4
        (60, 24.0, 1.0),  # C4
        (60, 25.0, 1.0),  # C4
        (62, 26.0, 1.0),  # D4
        (64, 27.0, 1.0),  # E4
        (62, 28.0, 1.5),  # D4
        (60, 29.5, 0.5),  # C4
        (60, 30.0, 2.0),  # C4
    ]


def build_mary_had_a_little_lamb() -> List[Tuple[int, float, float]]:
    """Mary Had a Little Lamb (32 beats, 26 notes)."""
    return [
        (64, 0.0, 1.0),   # E4
        (62, 1.0, 1.0),   # D4
        (60, 2.0, 1.0),   # C4
        (62, 3.0, 1.0),   # D4
        (64, 4.0, 1.0),   # E4
        (64, 5.0, 1.0),   # E4
        (64, 6.0, 2.0),   # E4
        (62, 8.0, 1.0),   # D4
        (62, 9.0, 1.0),   # D4
        (62, 10.0, 2.0),  # D4
        (64, 12.0, 1.0),  # E4
        (67, 13.0, 1.0),  # G4
        (67, 14.0, 2.0),  # G4
        (64, 16.0, 1.0),  # E4
        (62, 17.0, 1.0),  # D4
        (60, 18.0, 1.0),  # C4
        (62, 19.0, 1.0),  # D4
        (64, 20.0, 1.0),  # E4
        (64, 21.0, 1.0),  # E4
        (64, 22.0, 1.0),  # E4
        (64, 23.0, 1.0),  # E4
        (62, 24.0, 1.0),  # D4
        (62, 25.0, 1.0),  # D4
        (64, 26.0, 1.0),  # E4
        (62, 27.0, 1.0),  # D4
        (60, 28.0, 4.0),  # C4
    ]


def build_jingle_bells() -> List[Tuple[int, float, float]]:
    """Jingle Bells Chorus (32 beats, 25 notes)."""
    return [
        (64, 0.0, 1.0),   # E4
        (64, 1.0, 1.0),   # E4
        (64, 2.0, 2.0),   # E4
        (64, 4.0, 1.0),   # E4
        (64, 5.0, 1.0),   # E4
        (64, 6.0, 2.0),   # E4
        (64, 8.0, 1.0),   # E4
        (67, 9.0, 1.0),   # G4
        (60, 10.0, 1.0),  # C4
        (62, 11.0, 1.0),  # D4
        (64, 12.0, 4.0),  # E4
        (65, 16.0, 1.0),  # F4
        (65, 17.0, 1.0),  # F4
        (65, 18.0, 1.0),  # F4
        (65, 19.0, 1.0),  # F4
        (65, 20.0, 1.0),  # F4
        (64, 21.0, 1.0),  # E4
        (64, 22.0, 1.0),  # E4
        (64, 23.0, 1.0),  # E4
        (64, 24.0, 1.0),  # E4
        (62, 25.0, 1.0),  # D4
        (62, 26.0, 1.0),  # D4
        (64, 27.0, 1.0),  # E4
        (62, 28.0, 2.0),  # D4
        (67, 30.0, 2.0),  # G4
    ]


def generate_all() -> None:
    data_real = Path("data/real")
    data_real.mkdir(parents=True, exist_ok=True)

    create_midi_file(data_real / "twinkle_twinkle.mid", 90.0, build_twinkle_twinkle())
    create_midi_file(data_real / "ode_to_joy.mid", 100.0, build_ode_to_joy())
    create_midi_file(data_real / "mary_had_a_little_lamb.mid", 95.0, build_mary_had_a_little_lamb())
    create_midi_file(data_real / "jingle_bells.mid", 100.0, build_jingle_bells())


if __name__ == "__main__":
    generate_all()
