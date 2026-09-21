"""Evaluate a trained multi-key PPO model on real MIDI pieces in a folder."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from pianorl.eval import (
    PPOPlayer,
    MultiKeyEpisodeCounters,
    compute_multikey_metrics,
    evaluate_multikey_real_piece,
)
from pianorl.score import Score, load_score


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained multi-key PPO model on real MIDI files."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained multi-key PPO model .zip file.",
    )
    parser.add_argument(
        "--midi-dir",
        type=str,
        default="data/real",
        help="Folder containing MIDI files to evaluate (default: data/real).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    midi_dir = Path(args.midi_dir)
    if not midi_dir.exists():
        raise FileNotFoundError(f"MIDI directory not found at '{midi_dir}'")

    midi_files = sorted(list(midi_dir.glob("*.mid*")))
    if not midi_files:
        print(f"No MIDI files found in '{midi_dir}'.")
        return

    print(f"Loading {model_path.name}...")
    player = PPOPlayer(model_path)

    print(f"Evaluating {len(midi_files)} real piece(s) in {midi_dir}...\n")
    print("=" * 105)
    print(
        f"{'Piece Name':<28} | {'Notes':<6} | {'Chords':<7} | "
        f"{'Precision':<9} | {'Recall':<8} | {'F1':<7} | {'Exact Rate':<10} | {'Mean Reward'}"
    )
    print("-" * 105)

    all_counters: List[MultiKeyEpisodeCounters] = []

    for f in midi_files:
        score = load_score(f)
        counter, n_chords = evaluate_multikey_real_piece(player, score)
        all_counters.append(counter)
        m = compute_multikey_metrics(counter, num_pieces=1)

        print(
            f"{f.stem[:28]:<28} | {m.total_notes:<6} | {n_chords:<7} | "
            f"{m.precision:<9.3f} | {m.recall:<8.3f} | {m.f1:<7.3f} | {m.exact_rate:<10.3f} | {m.mean_reward:+.2f}"
        )

    print("-" * 105)
    total_counter = MultiKeyEpisodeCounters()
    for c in all_counters:
        total_counter = total_counter + c
    total_m = compute_multikey_metrics(total_counter, num_pieces=len(all_counters))
    print(
        f"{'Overall (Average)':<28} | {total_m.total_notes:<6} | {total_counter.chord_steps_total:<7} | "
        f"{total_m.precision:<9.3f} | {total_m.recall:<8.3f} | {total_m.f1:<7.3f} | {total_m.exact_rate:<10.3f} | {total_m.mean_reward:+.2f}"
    )
    print("=" * 105)


if __name__ == "__main__":
    main()
