"""Evaluate a trained multi-key PPO model on real MIDI pieces in a folder."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from pianorl.env import MultiKeyPianoEnv, RewardConfig
from pianorl.eval import PPOPlayer, MultiKeyEpisodeCounters, compute_multikey_metrics
from pianorl.score import Score, load_score, NUM_PIANO_KEYS


def evaluate_multikey_real_piece(
    player,
    score: Score,
    seed: int = 42,
) -> Tuple[MultiKeyEpisodeCounters, int]:
    """Step a multi-key player through a real MIDI piece in MultiKeyPianoEnv."""
    standard_rewards = RewardConfig(
        hit_exact=1.0,
        hit_off_by_one=0.5,
        wrong_press=-0.5,
        miss=-1.0,
    )
    env = MultiKeyPianoEnv(scores=[score], seed=seed, reward_config=standard_rewards)
    obs, info = env.reset(options={"piece_index": 0})

    # Count chord steps (steps with >= 2 target notes)
    step_target_counts: Dict[int, int] = {}
    for t in env.targets:
        s = t["start_step"]
        step_target_counts[s] = step_target_counts.get(s, 0) + 1

    chord_steps_total = sum(1 for cnt in step_target_counts.values() if cnt >= 2)
    chord_steps_exact = 0

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        current_step = env.current_step
        action = player.act(obs)
        act_arr = np.asarray(action)

        if step_target_counts.get(current_step, 0) >= 2:
            due_pitches = set(t["pitch"] for t in env.targets if t["start_step"] == current_step)
            struck_pitches = set(21 + idx for idx in range(NUM_PIANO_KEYS) if act_arr[idx] > 0)
            if struck_pitches == due_pitches:
                chord_steps_exact += 1

        obs, reward, terminated, truncated, info = env.step(act_arr)
        total_reward += reward

    counter = MultiKeyEpisodeCounters(
        hits_exact=info["hits_exact"],
        hits_off_by_one=info["hits_off_by_one"],
        wrong_presses=info["wrong_presses"],
        missed_notes=info["missed_notes"],
        total_notes=info["total_notes"],
        total_reward=total_reward,
        chord_steps_total=chord_steps_total,
        chord_steps_exact=chord_steps_exact,
    )
    return counter, chord_steps_total


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
