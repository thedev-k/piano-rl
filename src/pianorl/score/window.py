import warnings
import numpy as np

from .score import Score, pitch_to_note_name

# Standard 88-key piano range: A0 (pitch 21) to C8 (pitch 108)
MIN_PIANO_PITCH = 21
MAX_PIANO_PITCH = 108
NUM_PIANO_KEYS = 88


class ScoreWindow:
    """Provides a sliding lookahead window into a musical score.

    The window gives the learning agent visibility into only the upcoming
    N beats, converted into discrete time steps.

    Attributes:
        score: The Score object to extract windows from.
        n_beats: How many upcoming beats the agent can see (default: 4).
        steps_per_beat: Resolution in steps per beat (default: 4 for 16th-notes).
        num_slots: Total time slots in the window (n_beats * steps_per_beat).
    """

    def __init__(
        self,
        score: Score,
        n_beats: int = 4,
        steps_per_beat: int = 4,
    ):
        if n_beats <= 0:
            raise ValueError(f"n_beats must be positive, got {n_beats}")
        if steps_per_beat <= 0:
            raise ValueError(f"steps_per_beat must be positive, got {steps_per_beat}")

        self.score = score
        self.n_beats = n_beats
        self.steps_per_beat = steps_per_beat
        self.num_slots = n_beats * steps_per_beat
        self._warned_pitches = set()

        # Pre-index valid notes by start_step for high performance on large scores
        self._notes_by_step = {}
        self._out_of_range_notes = []
        self._max_dur_steps = 1
        for note in self.score.notes:
            if note.pitch < MIN_PIANO_PITCH or note.pitch > MAX_PIANO_PITCH:
                self._out_of_range_notes.append(note)
                continue

            row = note.pitch - MIN_PIANO_PITCH
            start_step = int(round(note.start_beat * self.steps_per_beat))
            dur_steps = max(1, int(round(note.duration_beats * self.steps_per_beat)))
            end_step = start_step + dur_steps
            if dur_steps > self._max_dur_steps:
                self._max_dur_steps = dur_steps
            self._notes_by_step.setdefault(start_step, []).append((row, start_step, end_step))

    def total_steps(self) -> int:
        """Return how many time steps the entire piece lasts."""
        if not self.score.notes:
            return 0
        max_step = 0
        for step_notes in self._notes_by_step.values():
            for _, _, end_step in step_notes:
                if end_step > max_step:
                    max_step = end_step
        return max_step

    def get_window(self, current_step: int) -> np.ndarray:
        """Extract the sliding lookahead window starting at current_step.

        Args:
            current_step: Integer step from the start of the score.

        Returns:
            np.ndarray: Array of shape (2, 88, num_slots) with dtype float32:
                - Channel 0: Note onset (1.0 where a note begins, else 0.0)
                - Channel 1: Note sounding (1.0 while note is held, else 0.0)
                - Rows: 88 piano keys (row 0 is pitch 21 / A0, row 87 is pitch 108 / C8)
                - Columns: Upcoming time slots [current_step, current_step + num_slots)
        """
        # Always return fixed shape (2, 88, num_slots)
        window = np.zeros((2, NUM_PIANO_KEYS, self.num_slots), dtype=np.float32)

        for note in self._out_of_range_notes:
            if note.pitch not in self._warned_pitches:
                warning_msg = (
                    f"Note pitch {note.pitch} ({note.note_name}) is outside "
                    f"the 88-key piano range ({MIN_PIANO_PITCH}-{MAX_PIANO_PITCH}). Ignoring note."
                )
                print(f"WARNING: {warning_msg}")
                warnings.warn(warning_msg, UserWarning, stacklevel=2)
                self._warned_pitches.add(note.pitch)

        window_start = current_step
        window_end = current_step + self.num_slots

        # Only inspect steps that could overlap with [window_start, window_end)
        min_start = max(0, window_start - self._max_dur_steps)
        for s in range(min_start, window_end):
            step_notes = self._notes_by_step.get(s)
            if not step_notes:
                continue
            for row, start_step, end_step in step_notes:
                if end_step <= window_start or start_step >= window_end:
                    continue

                # Populate time slots inside this window
                for slot in range(self.num_slots):
                    abs_step = window_start + slot

                    # Channel 0: Onset (note starts at this exact step)
                    if abs_step == start_step:
                        window[0, row, slot] = 1.0

                    # Channel 1: Sounding (note is active/held at this step)
                    if start_step <= abs_step < end_step:
                        window[1, row, slot] = 1.0

        return window
