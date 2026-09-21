"""Evaluation script for trained multi-key PPO models against baseline players."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from pianorl.agent import PerfectMultiPlayer
from pianorl.eval import (
    PPOPlayer,
    SilentMultiPlayer,
    RandomMultiPlayer,
    MultiKeyEvaluationMetrics,
    evaluate_multi_player,
)
from pianorl.score import load_score
from pianorl.data import normalize_level_tag


def print_multikey_results_table(
    results: Dict[str, MultiKeyEvaluationMetrics],
    player_name: str,
    split_name: str,
) -> None:
    print("=" * 105)
    print(f"Multi-Key Evaluation Results: Player = {player_name.upper()} | Split = {split_name.upper()}")
    print("=" * 105)
    print(
        f"{'Level':<10} | {'Pieces':<6} | {'Notes':<6} | "
        f"{'Precision':<9} | {'Recall':<8} | {'F1 Score':<8} | {'Exact Rate':<10} | {'Chord Match':<11} | {'Mean Reward'}"
    )
    print("-" * 105)

    for name, m in results.items():
        if name == "Overall":
            print("-" * 105)
        print(
            f"{name:<10} | {m.num_pieces:<6} | {m.total_notes:<6} | "
            f"{m.precision:<9.3f} | {m.recall:<8.3f} | {m.f1:<8.3f} | {m.exact_rate:<10.3f} | "
            f"{m.chord_exact_rate * 100:<10.1f}% | {m.mean_reward:+.2f}"
        )
    print("=" * 105 + "\n")


def print_baseline_comparison(
    player_results: Dict[str, MultiKeyEvaluationMetrics],
    split_name: str,
) -> None:
    print("=" * 95)
    print(f"BENCHMARK COMPARISON ACROSS PLAYERS (Overall on {split_name.upper()} split)")
    print("=" * 95)
    print(
        f"{'Player':<20} | {'Precision':<9} | {'Recall':<8} | {'F1 Score':<8} | "
        f"{'Exact Rate':<10} | {'Chord Match':<11} | {'Mean Reward'}"
    )
    print("-" * 95)

    for name, m in player_results.items():
        print(
            f"{name:<20} | {m.precision:<9.3f} | {m.recall:<8.3f} | {m.f1:<8.3f} | "
            f"{m.exact_rate:<10.3f} | {m.chord_exact_rate * 100:<10.1f}% | {m.mean_reward:+.2f}"
        )
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-key PPO model and baseline players on polyphonic pieces."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained multi-key PPO model .zip file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="heldout",
        choices=["heldout", "train"],
        help="Dataset split to evaluate (default: heldout).",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help="Specific levels to evaluate (e.g. 1M or 1 2 3). Defaults to all levels.",
    )
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        default=True,
        help="Compare model against Perfect, Silent, and Random baseline players (default: True).",
    )

    args = parser.parse_args()

    manifest_file = Path("data/manifest_multikey.json")
    if not manifest_file.exists():
        raise FileNotFoundError(
            "data/manifest_multikey.json not found. Run scripts/generate_multikey_dataset.py first."
        )

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_items = [item for item in manifest if item["split"] == args.split]
    if args.levels:
        target_lvls = {normalize_level_tag(l) for l in args.levels}
        split_items = [
            it for it in split_items if normalize_level_tag(it["level"]) in target_lvls
        ]

    if not split_items:
        raise ValueError(f"No pieces found for split='{args.split}' and levels={args.levels}")

    print(f"Loading {len(split_items)} {args.split} pieces for multi-key evaluation...")
    data_dir = Path("data")
    scores_list = [load_score(data_dir / item["filename"]) for item in split_items]

    baseline_summaries: Dict[str, MultiKeyEvaluationMetrics] = {}

    # 1. Evaluate trained model if provided
    if args.model_path:
        model_file = Path(args.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at '{model_file}'")
        print(f"\nEvaluating PPO model: {model_file.name}...")
        model_player = PPOPlayer(model_file)
        model_results = evaluate_multi_player(model_player, scores_list, split_items)
        print_multikey_results_table(model_results, f"PPO ({model_file.name})", args.split)
        baseline_summaries[f"PPO ({model_file.stem})"] = model_results["Overall"]

    # 2. Evaluate baselines if requested
    if args.compare_baselines:
        print("Evaluating Perfect Sight-Reading Baseline (PerfectMultiPlayer)...")
        perfect_player = PerfectMultiPlayer()
        perf_results = evaluate_multi_player(perfect_player, scores_list, split_items)
        baseline_summaries["Perfect (Oracle)"] = perf_results["Overall"]

        print("Evaluating Silent Baseline (SilentMultiPlayer)...")
        silent_player = SilentMultiPlayer()
        silent_results = evaluate_multi_player(silent_player, scores_list, split_items)
        baseline_summaries["Silent (Do Nothing)"] = silent_results["Overall"]

        print("Evaluating Random Baseline (RandomMultiPlayer, p=5%)...")
        random_player = RandomMultiPlayer(prob=0.05, seed=42)
        rand_results = evaluate_multi_player(random_player, scores_list, split_items)
        baseline_summaries["Random (p=5%)"] = rand_results["Overall"]

        print_baseline_comparison(baseline_summaries, args.split)


if __name__ == "__main__":
    main()
