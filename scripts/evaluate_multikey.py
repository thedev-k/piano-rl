"""Evaluation script for trained multi-key PPO models against baseline players."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def extract_checkpoint_step(path: Path) -> int:
    """Extract step count from checkpoint filename for chronological sorting."""
    import re
    stem = path.stem
    match = re.search(r"(\d+)_steps", stem)
    if match:
        return int(match.group(1))
    match_any_num = re.search(r"(\d+)", stem)
    if match_any_num:
        return int(match_any_num.group(1))
    return 999_999_999  # Place final.zip or unnumbered at the end


def print_checkpoint_progression_table(
    records: List[Dict[str, Any]],
    levels: List[str],
    split_name: str,
) -> None:
    """Print a clean consolidated trend table comparing multiple checkpoints across levels."""
    levels = sorted(levels)
    col_w = 12
    hdr = f"{'Checkpoint / Step':<30} | " + " | ".join(f"{f'{lvl} Chord':^{col_w}}" for lvl in levels)
    hdr += f" | {'Overall F1':^10} | {'Exact Rate':^10} | {'Chord Match':^11} | {'Mean Reward':^11}"
    border = "=" * len(hdr)
    sub_border = "-" * len(hdr)

    print("\n" + border)
    print(f"MULTI-CHECKPOINT PROGRESSION SUMMARY (Split = {split_name.upper()})")
    print(border)
    print(hdr)
    print(sub_border)

    for rec in records:
        name = rec["name"][:30]
        lvl_chords = []
        for lvl in levels:
            norm_lvl = normalize_level_tag(lvl)
            matched_key = None
            for k in rec["levels"]:
                clean_k = k.replace("Level", "").strip().upper()
                if not clean_k.endswith("M"):
                    clean_k += "M"
                if clean_k == norm_lvl or k == lvl or k == f"Level {lvl}":
                    matched_key = k
                    break
            if matched_key is not None:
                chord_pct = f"{rec['levels'][matched_key].chord_exact_rate * 100:.1f}%"
            else:
                chord_pct = "N/A"
            lvl_chords.append(f"{chord_pct:^{col_w}}")
        ov = rec["overall"]
        row = f"{name:<30} | " + " | ".join(lvl_chords)
        row += f" | {ov.f1:^10.3f} | {ov.exact_rate:^10.3f} | {ov.chord_exact_rate * 100:^10.1f}% | {ov.mean_reward:^+11.2f}"
        print(row)
    print(border + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-key PPO model and baseline players on polyphonic pieces."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to single trained multi-key PPO model .zip file.",
    )
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=None,
        help="List of model checkpoint .zip files to evaluate in sequence.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing multiple checkpoint .zip files to evaluate.",
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
        action=argparse.BooleanOptionalAction,
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

    # Collect model paths
    candidate_paths: List[Path] = []
    if args.model_path:
        candidate_paths.append(Path(args.model_path))
    if args.model_paths:
        for p in args.model_paths:
            candidate_paths.append(Path(p))
    if args.checkpoint_dir:
        cdir = Path(args.checkpoint_dir)
        if not cdir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: '{cdir}'")
        found = list(cdir.glob("*.zip"))
        if not found:
            raise FileNotFoundError(f"No .zip checkpoints found in '{cdir}'")
        candidate_paths.extend(found)

    # Deduplicate while preserving order, then sort by step count
    unique_paths: List[Path] = []
    seen = set()
    for p in candidate_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_paths.append(p)

    if len(unique_paths) > 1:
        unique_paths.sort(key=extract_checkpoint_step)

    progression_records: List[Dict[str, Any]] = []

    # 1. Evaluate trained models if provided
    for model_file in unique_paths:
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at '{model_file}'")
        print(f"\nEvaluating PPO model: {model_file.name}...")
        model_player = PPOPlayer(model_file)
        model_results = evaluate_multi_player(model_player, scores_list, split_items)
        print_multikey_results_table(model_results, f"PPO ({model_file.name})", args.split)
        baseline_summaries[f"PPO ({model_file.stem})"] = model_results["Overall"]
        progression_records.append(
            {
                "name": model_file.stem,
                "overall": model_results["Overall"],
                "levels": {k: v for k, v in model_results.items() if k != "Overall"},
            }
        )

    # Print consolidated multi-checkpoint summary table if multiple checkpoints evaluated
    if len(progression_records) > 1:
        eval_levels = sorted(list({normalize_level_tag(it["level"]) for it in split_items}))
        print_checkpoint_progression_table(progression_records, eval_levels, args.split)

    # 2. Evaluate baselines if requested
    if args.compare_baselines and not unique_paths:
        # If no models were passed, run baselines
        run_baselines = True
    elif args.compare_baselines:
        run_baselines = True
    else:
        run_baselines = False

    if run_baselines:
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
