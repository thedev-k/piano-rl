import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from pianorl.env import PianoFreeKeysEnv, RewardConfig
from pianorl.eval import (
    DoNothingPlayer,
    EpisodeCounters,
    EvaluationMetrics,
    RandomPlayer,
    RuleBasedPlayer,
    aggregate_and_compute,
)
from pianorl.score import load_score


def evaluate_player(
    player,
    scores_list: List,
    split_items: List[dict],
) -> Dict[str, EvaluationMetrics]:
    """Evaluate any player object implementing `act(observation)` on a set of scores.

    Args:
        player: Object with an `act(observation) -> int` method.
        scores_list: List of loaded Score objects.
        split_items: List of manifest items matching the scores.

    Returns:
        Dict mapping level string (e.g. 'Level 1', 'Overall') to EvaluationMetrics.
    """
    # IMPORTANT: Evaluation rewards must ALWAYS use the standard default numbers
    # (hit_exact=+1.0, hit_off_by_one=+0.5, wrong_press=-0.5, miss=-1.0).
    # This ensures that Mean Reward remains directly comparable across runs,
    # even when different models were trained with customized reward weights.
    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = PianoFreeKeysEnv(scores=scores_list, seed=42, reward_config=standard_rewards)

    level_counters: Dict[int, List[EpisodeCounters]] = {1: [], 2: [], 3: [], 4: []}
    all_counters: List[EpisodeCounters] = []

    for i, item in enumerate(split_items):
        obs, info = env.reset(options={"piece_index": i})
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = player.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        ep_counter = EpisodeCounters(
            hits_exact=info["hits_exact"],
            hits_off_by_one=info["hits_off_by_one"],
            wrong_presses=info["wrong_presses"],
            missed_notes=info["missed_notes"],
            total_notes=info["total_notes"],
            total_reward=total_reward,
        )

        level = item["level"]
        level_counters[level].append(ep_counter)
        all_counters.append(ep_counter)

    results: Dict[str, EvaluationMetrics] = {}
    for lvl in sorted(level_counters.keys()):
        if level_counters[lvl]:
            results[f"Level {lvl}"] = aggregate_and_compute(level_counters[lvl])

    results["Overall"] = aggregate_and_compute(all_counters)
    return results


def print_results_table(
    results: Dict[str, EvaluationMetrics],
    player_name: str,
    split_name: str,
) -> None:
    print("=" * 87)
    print(f"Evaluation Results: Player = {player_name.upper()} | Split = {split_name.upper()}")
    print("=" * 87)
    print(
        f"{'Level':<10} | {'Pieces':<6} | {'Notes':<6} | "
        f"{'Precision':<9} | {'Recall':<8} | {'F1 Score':<8} | {'Exact Rate':<10} | {'Mean Reward'}"
    )
    print("-" * 87)

    for name, m in results.items():
        if name == "Overall":
            print("-" * 87)
        print(
            f"{name:<10} | {m.num_pieces:<6} | {m.total_notes:<6} | "
            f"{m.precision:<9.4f} | {m.recall:<8.4f} | {m.f1:<8.4f} | "
            f"{m.exact_rate:<10.4f} | {m.mean_reward:+.2f}"
        )
    print("=" * 87)


def save_results_csv(
    results: Dict[str, EvaluationMetrics],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Level",
                "Pieces",
                "Total_Notes",
                "Precision",
                "Recall",
                "F1_Score",
                "Exact_Rate",
                "Mean_Reward",
            ]
        )
        for name, m in results.items():
            writer.writerow(
                [
                    name,
                    m.num_pieces,
                    m.total_notes,
                    f"{m.precision:.4f}",
                    f"{m.recall:.4f}",
                    f"{m.f1:.4f}",
                    f"{m.exact_rate:.4f}",
                    f"{m.mean_reward:.2f}",
                ]
            )
    print(f"Saved CSV table to: {csv_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a player on the PianoFreeKeysEnv dataset.")
    parser.add_argument(
        "--player",
        type=str,
        default="rule_based",
        choices=["random", "silent", "rule_based", "ppo"],
        help="Player to evaluate (random, silent, rule_based, or ppo)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="heldout",
        choices=["train", "heldout"],
        help="Dataset split to evaluate on (train or heldout)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained PPO model zip file (required if --player ppo)",
    )
    args = parser.parse_args()

    manifest_file = Path("data/manifest.json")
    if not manifest_file.exists():
        raise FileNotFoundError("data/manifest.json not found. Run scripts/generate_dataset.py first.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter items by split
    split_items = [item for item in manifest if item["split"] == args.split]
    if not split_items:
        raise ValueError(f"No pieces found in manifest for split '{args.split}'")

    data_dir = Path("data")
    print(f"Loading {len(split_items)} {args.split} pieces...")
    scores_list = [load_score(data_dir / item["filename"]) for item in split_items]

    # Instantiate player
    if args.player == "random":
        player = RandomPlayer(seed=42)
        save_name = f"results_random_{args.split}.csv"
    elif args.player == "silent":
        player = DoNothingPlayer()
        save_name = f"results_silent_{args.split}.csv"
    elif args.player == "rule_based":
        player = RuleBasedPlayer()
        save_name = f"results_rule_based_{args.split}.csv"
    elif args.player == "ppo":
        if not args.model_path:
            raise ValueError("--model-path is required when --player is ppo")
        model_p = Path(args.model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Trained model not found at '{model_p}'")
        from pianorl.eval import PPOPlayer
        player = PPOPlayer(model_p)
        save_name = f"results_ppo_{model_p.stem}_{args.split}.csv"
    else:
        raise ValueError(f"Unknown player {args.player}")

    print(f"Running evaluation with {args.player.upper()} on {args.split.upper()} split...")
    results = evaluate_player(player, scores_list, split_items)

    print_results_table(results, args.player, args.split)

    csv_path = Path("outputs") / save_name
    save_results_csv(results, csv_path)



if __name__ == "__main__":
    main()
