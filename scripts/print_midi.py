import sys
from pathlib import Path
from pianorl.score import load_score


def print_score_table(file_path: str) -> None:
    path = Path(file_path)
    score = load_score(path)

    print("=" * 65)
    print(f"File: {path}")
    print(f"Tempo: {score.tempo_bpm} BPM | Total Notes: {len(score)} | Total Length: {score.total_beats} beats")
    print("=" * 65)
    print(f"{'#':<4} | {'Pitch':<6} | {'Note':<6} | {'Start (beat)':<14} | {'Duration (beats)':<16} | {'End (beat)':<10}")
    print("-" * 65)

    for i, note in enumerate(score.notes, start=1):
        print(
            f"{i:<4} | {note.pitch:<6} | {note.note_name:<6} | "
            f"{note.start_beat:<14.2f} | {note.duration_beats:<16.2f} | {note.end_beat:<10.2f}"
        )

    print("=" * 65)


def main():
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = "data/examples/c_major_scale.mid"

    try:
        print_score_table(target_path)
    except Exception as err:
        print(f"Error loading MIDI file: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
