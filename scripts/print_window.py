import sys
from pathlib import Path

from pianorl.score import (
    load_score,
    ScoreWindow,
    MIN_PIANO_PITCH,
    NUM_PIANO_KEYS,
    pitch_to_note_name,
)


def print_window_ascii(
    file_path: str,
    current_step: int = 0,
    n_beats: int = 4,
    steps_per_beat: int = 4,
) -> None:
    path = Path(file_path)
    score = load_score(path)
    window_engine = ScoreWindow(score, n_beats=n_beats, steps_per_beat=steps_per_beat)

    window = window_engine.get_window(current_step)
    slots = window_engine.num_slots
    total_steps = window_engine.total_steps()

    print("=" * 65)
    print(f"File: {path}")
    print(
        f"Window at Step: {current_step} (Beat {current_step / steps_per_beat:.2f}) | "
        f"Length: {n_beats} beats ({slots} slots) | Total Piece Steps: {total_steps}"
    )
    print(f"Legend: '#' = note starts, '=' = note held, '.' = empty")
    print("=" * 65)

    # Find keys with any activity in this window
    active_rows = [r for r in range(NUM_PIANO_KEYS) if window[:, r, :].any()]

    # Print slot numbers header
    slot_header_tens = "".join(str((s // 10) if s >= 10 else " ") for s in range(slots))
    slot_header_ones = "".join(str(s % 10) for s in range(slots))
    print(f"{'Key':<9} | {slot_header_tens}  (Tens)")
    print(f"{'(Pitch)':<9} | {slot_header_ones}  (Slot)")
    print("-" * (12 + slots))

    if not active_rows:
        print(f"{'--':<9} | {'.' * slots}  (No active notes in this window)")
    else:
        # Highest pitch at the top
        for row in sorted(active_rows, reverse=True):
            pitch = MIN_PIANO_PITCH + row
            note_str = f"{pitch_to_note_name(pitch)} ({pitch})"
            chars = []
            for s in range(slots):
                if window[0, row, s] == 1.0:
                    chars.append("#")
                elif window[1, row, s] == 1.0:
                    chars.append("=")
                else:
                    chars.append(".")
            print(f"{note_str:<9} | {''.join(chars)}")

    print("=" * 65)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/examples/c_major_scale.mid"
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_beats = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    try:
        print_window_ascii(path, current_step=step, n_beats=n_beats)
    except Exception as err:
        print(f"Error displaying window: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
