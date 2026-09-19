import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pianorl.score import load_score, pitch_to_note_name


def plot_score(midi_path: str | Path, output_path: str | Path | None = None) -> Path:
    input_file = Path(midi_path)
    score = load_score(input_file)

    if output_path is None:
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_file = outputs_dir / f"{input_file.stem}_piano_roll.png"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    pitches = [n.pitch for n in score.notes]
    min_pitch = min(pitches) if pitches else 60
    max_pitch = max(pitches) if pitches else 72

    # Draw each note as a horizontal bar
    for note in score.notes:
        rect = patches.Rectangle(
            (note.start_beat, note.pitch - 0.4),
            note.duration_beats,
            0.8,
            linewidth=1,
            edgecolor="#1E3A8A",
            facecolor="#3B82F6",
            alpha=0.85,
        )
        ax.add_patch(rect)

    # Configure axes
    ax.set_xlim(-0.5, max(score.total_beats + 0.5, 4.0))
    ax.set_ylim(min_pitch - 1.5, max_pitch + 1.5)

    # Set vertical ticks to note pitches and names
    y_ticks = list(range(min_pitch - 1, max_pitch + 2))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{pitch_to_note_name(p)} ({p})" if 21 <= p <= 108 else "" for p in y_ticks], fontsize=8)

    ax.set_xlabel("Time (beats)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Piano Key / Pitch", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Piano Roll: {input_file.name} | Tempo: {score.tempo_bpm:.0f} BPM | {len(score)} Notes",
        fontsize=12,
        fontweight="bold",
    )

    # Musical grid lines
    ax.grid(True, which="both", axis="x", color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.grid(True, which="both", axis="y", color="#F1F5F9", linestyle="-", linewidth=0.5)

    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved piano-roll plot: {output_file}")
    return output_file


def main():
    if len(sys.argv) > 1:
        midi_path = sys.argv[1]
    else:
        midi_path = "data/examples/c_major_scale.mid"

    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    plot_score(midi_path, out_path)


if __name__ == "__main__":
    main()
