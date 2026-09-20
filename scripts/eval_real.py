"""Evaluate a trained Piano-RL model on real MIDI melodies in a folder.

Loads every .mid file in a folder (default: data/real/), runs the model using
the fair PPOPlayer interface, computes performance metrics, checks for polyphony/chords,
and saves visual piano-roll comparisons to outputs/real/.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pianorl.env import PianoFreeKeysEnv, RewardConfig
from pianorl.eval import PPOPlayer, compute_metrics, EpisodeCounters
from pianorl.score import Score, load_score, pitch_to_note_name


def check_chords_or_simultaneous_notes(score: Score) -> bool:
    """Check whether a score contains multiple note onsets at the exact same beat."""
    start_steps = [round(note.start_beat * 4) for note in score.notes]
    return len(start_steps) != len(set(start_steps))


def plot_piano_roll_comparison(
    score: Score,
    model_presses: List[dict],
    piece_name: str,
    output_path: Path,
    precision: float,
    recall: float,
    f1: float,
) -> None:
    """Save a piano-roll image comparing target notes to model key presses."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))

    # 1. Target notes (horizontal bars)
    for note in score.notes:
        rect = patches.Rectangle(
            (note.start_beat, note.pitch - 0.4),
            note.duration_beats,
            0.8,
            linewidth=1.2,
            edgecolor="#1f77b4",
            facecolor="#aec7e8",
            alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            note.start_beat + 0.08,
            note.pitch,
            note.note_name,
            va="center",
            ha="left",
            fontsize=8,
            color="#08306b",
            fontweight="bold",
            zorder=4,
        )

    # 2. Model presses
    hit_beats = [p["beat"] for p in model_presses if p["result"] in ("exact", "off_by_one")]
    hit_pitches = [p["pitch"] for p in model_presses if p["result"] in ("exact", "off_by_one")]

    wrong_beats = [p["beat"] for p in model_presses if p["result"] == "wrong"]
    wrong_pitches = [p["pitch"] for p in model_presses if p["result"] == "wrong"]

    if hit_beats:
        ax.scatter(
            hit_beats,
            hit_pitches,
            color="#2ca02c",
            marker="o",
            s=80,
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
            label="Model Hit (exact / off-by-one)",
        )
    if wrong_beats:
        ax.scatter(
            wrong_beats,
            wrong_pitches,
            color="#d62728",
            marker="x",
            s=90,
            linewidths=2.0,
            zorder=7,
            label="Model Wrong Press",
        )

    # Y-axis limits and labels
    all_pitches = [n.pitch for n in score.notes] + [p["pitch"] for p in model_presses]
    if all_pitches:
        min_pitch = min(all_pitches) - 2
        max_pitch = max(all_pitches) + 2
    else:
        min_pitch, max_pitch = 60, 72

    ax.set_ylim(min_pitch - 0.5, max_pitch + 0.5)
    ax.set_yticks(range(min_pitch, max_pitch + 1))
    ax.set_yticklabels([f"{pitch_to_note_name(p)} ({p})" for p in range(min_pitch, max_pitch + 1)])

    max_beat = max(score.total_beats, max([p["beat"] for p in model_presses], default=0.0))
    ax.set_xlim(-0.5, max_beat + 1.0)
    ax.set_xlabel("Time (beats)", fontsize=11)
    ax.set_ylabel("Piano Key / Pitch", fontsize=11)

    title_clean = piece_name.replace("_", " ").title()
    ax.set_title(
        f"Piano Roll: {title_clean} | Notes: {len(score)} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="upper right", framealpha=0.9)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_single_piece(
    score: Score,
    player: PPOPlayer,
) -> Tuple[EpisodeCounters, List[dict]]:
    """Play a single score start-to-finish with a PPO player and record actions."""
    # Standard evaluation rewards (+1.0, +0.5, -0.5, -1.0)
    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = PianoFreeKeysEnv(scores=[score], seed=42, reward_config=standard_rewards)
    obs, info = env.reset(options={"piece_index": 0})

    total_reward = 0.0
    terminated = False
    truncated = False

    prev_hits_exact = 0
    prev_hits_off = 0
    prev_wrong = 0

    model_presses: List[dict] = []

    while not (terminated or truncated):
        step_now = env.current_step
        action = player.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if action > 0:
            pressed_pitch = 20 + action
            if info["hits_exact"] > prev_hits_exact:
                res = "exact"
            elif info["hits_off_by_one"] > prev_hits_off:
                res = "off_by_one"
            else:
                res = "wrong"

            model_presses.append(
                {
                    "step": step_now,
                    "beat": step_now / env.steps_per_beat,
                    "pitch": pressed_pitch,
                    "result": res,
                }
            )

        prev_hits_exact = info["hits_exact"]
        prev_hits_off = info["hits_off_by_one"]
        prev_wrong = info["wrong_presses"]

    counters = EpisodeCounters(
        hits_exact=info["hits_exact"],
        hits_off_by_one=info["hits_off_by_one"],
        wrong_presses=info["wrong_presses"],
        missed_notes=info["missed_notes"],
        total_notes=info["total_notes"],
        total_reward=total_reward,
    )
    return counters, model_presses


def evaluate_real_folder(
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
) -> Dict[str, dict]:
    """Evaluate all MIDI files in data_dir with the model and save piano roll images."""
    data_dir = Path(data_dir)
    midi_files = sorted(list(data_dir.glob("*.mid")) + list(data_dir.glob("*.midi")))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in '{data_dir}'")

    print(f"Loading PPO player from '{model_path}'...")
    player = PPOPlayer(model_path)

    results = {}
    total_counters = EpisodeCounters()
    num_pieces = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating {len(midi_files)} real piece(s) from '{data_dir}'...\n")

    for midi_file in midi_files:
        score = load_score(midi_file)
        has_chords = check_chords_or_simultaneous_notes(score)

        counters, model_presses = evaluate_single_piece(score, player)
        metrics = compute_metrics(counters, num_pieces=1)

        # Plot piano roll
        img_path = output_dir / f"{midi_file.stem}_pianoroll.png"
        plot_piano_roll_comparison(
            score=score,
            model_presses=model_presses,
            piece_name=midi_file.stem,
            output_path=img_path,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1,
        )

        results[midi_file.name] = {
            "notes": counters.total_notes,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "exact_rate": metrics.exact_rate,
            "reward": counters.total_reward,
            "has_chords": has_chords,
            "img_path": img_path,
        }

        total_counters = total_counters + counters
        num_pieces += 1

    overall_metrics = compute_metrics(total_counters, num_pieces=num_pieces)
    results["Overall"] = {
        "notes": total_counters.total_notes,
        "precision": overall_metrics.precision,
        "recall": overall_metrics.recall,
        "f1": overall_metrics.f1,
        "exact_rate": overall_metrics.exact_rate,
        "reward": overall_metrics.mean_reward,
        "has_chords": any(r["has_chords"] for k, r in results.items() if k != "Overall"),
        "img_path": None,
    }

    return results


def print_real_evaluation_table(results: Dict[str, dict], model_path: Path, data_dir: Path) -> None:
    """Print a clean, formatted table of real melody evaluation results."""
    print("=" * 105)
    print(f"REAL MELODIES EVALUATION: Model = {model_path.name} | Folder = {data_dir}")
    print("=" * 105)
    print(
        f"{'Piece':<30} | {'Notes':<5} | {'Precision':<9} | {'Recall':<6} | "
        f"{'F1':<6} | {'Exact Rate':<10} | {'Reward':<7} | {'Polyphony / Chords'}"
    )
    print("-" * 105)

    has_any_chords = False
    for name, r in results.items():
        if name == "Overall":
            print("-" * 105)
            chords_str = "N/A"
        else:
            if r["has_chords"]:
                chords_str = "WARNING: Simultaneous Notes (Chords)"
                has_any_chords = True
            else:
                chords_str = "Single-line (OK)"

        print(
            f"{name:<30} | {r['notes']:<5} | {r['precision']:<9.3f} | {r['recall']:<6.3f} | "
            f"{r['f1']:<6.3f} | {r['exact_rate']:<10.3f} | {r['reward']:<+7.2f} | {chords_str}"
        )

    print("=" * 105)
    if has_any_chords:
        print(
            "\n* NOTICE: The Free-Keys piano agent can strike at most ONE key per step.\n"
            "  Pieces with chords will have overlapping notes that cannot be played simultaneously."
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Piano-RL agent on real MIDI melodies.")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained PPO model zip file (e.g. checkpoints/all_pitch_conv/final.zip)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/real",
        help="Directory containing real .mid melody files (default: data/real)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/real",
        help="Directory to save piano-roll visualization images (default: outputs/real)",
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    results = evaluate_real_folder(data_dir=data_dir, model_path=model_path, output_dir=output_dir)
    print_real_evaluation_table(results, model_path, data_dir)
    print(f"\nPiano-roll images saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
