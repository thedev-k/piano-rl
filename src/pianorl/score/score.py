from dataclasses import dataclass
from typing import List, Iterator

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_to_note_name(pitch: int) -> str:
    """Convert a MIDI pitch number (e.g., 60) to scientific note name (e.g., 'C4')."""
    if not (0 <= pitch <= 127):
        raise ValueError(f"Pitch must be between 0 and 127, got {pitch}")
    octave = (pitch // 12) - 1
    name = NOTE_NAMES[pitch % 12]
    return f"{name}{octave}"


@dataclass(frozen=True)
class NoteEvent:
    """Represents a single musical note played in a score.

    Attributes:
        pitch: MIDI note number (0 to 127, where 60 is Middle C).
        start_beat: Time when the note starts, measured in beats (>= 0).
        duration_beats: How long the note lasts, measured in beats (> 0).
    """
    pitch: int
    start_beat: float
    duration_beats: float

    def __post_init__(self):
        if not (0 <= self.pitch <= 127):
            raise ValueError(f"Pitch must be between 0 and 127, got {self.pitch}")
        if self.start_beat < 0:
            raise ValueError(f"start_beat must be non-negative, got {self.start_beat}")
        if self.duration_beats <= 0:
            raise ValueError(f"duration_beats must be strictly positive, got {self.duration_beats}")

    @property
    def note_name(self) -> str:
        """Return the note name, like 'C4' or 'F#5'."""
        return pitch_to_note_name(self.pitch)

    @property
    def end_beat(self) -> float:
        """Return the beat when the note stops sounding."""
        return self.start_beat + self.duration_beats


@dataclass
class Score:
    """Represents a complete musical score with note events and tempo.

    Attributes:
        notes: List of NoteEvent objects, sorted by start_beat.
        tempo_bpm: Tempo in beats per minute (BPM).
    """
    notes: List[NoteEvent]
    tempo_bpm: float = 120.0

    def __post_init__(self):
        if self.tempo_bpm <= 0:
            raise ValueError(f"tempo_bpm must be positive, got {self.tempo_bpm}")

    def __len__(self) -> int:
        return len(self.notes)

    def __iter__(self) -> Iterator[NoteEvent]:
        return iter(self.notes)

    def __getitem__(self, index: int) -> NoteEvent:
        return self.notes[index]

    @property
    def total_beats(self) -> float:
        """The beat timestamp when the last sounding note ends."""
        if not self.notes:
            return 0.0
        return max(note.end_beat for note in self.notes)
