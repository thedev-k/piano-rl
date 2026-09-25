"""Multi-Key Polyphonic Piano Reinforcement Learning Training Pipeline.

Trains a PPO agent with MultiKeyPitchConvPolicy on multi-key procedural pieces (Levels 1M to 8M)
using Gymnasium MultiBinary(88) action spaces.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pianorl.agent import MultiKeyPitchConvPolicy, MultiKeyTensorboardCallback
from pianorl.data import normalize_level_tag
from pianorl.env import MultiKeyPianoEnv, RewardConfig
from pianorl.score import Score, load_score


class PlainProgressCallback(BaseCallback):
    """Prints training progress in plain, friendly English at regular intervals."""

    def __init__(self, total_timesteps: int, print_freq: int = 5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.last_print_step = 0
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print(f"Training started! Target: {self.total_timesteps:,} timesteps.\n")

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        if current_step - self.last_print_step >= self.print_freq:
            elapsed = time.time() - self.start_time
            fps = current_step / elapsed if elapsed > 0 else 0
            percent = (current_step / self.total_timesteps) * 100
            remaining_steps = max(0, self.total_timesteps - current_step)
            eta_seconds = remaining_steps / fps if fps > 0 else 0
            eta_str = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60):02d}s"

            ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
            if ep_info_buffer and len(ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in ep_info_buffer]
                mean_r = sum(recent_rewards) / len(recent_rewards)
                reward_str = f"{mean_r:+.2f}"
            else:
                reward_str = "calculating..."

            print(
                f"[Progress] {current_step:,} / {self.total_timesteps:,} steps ({percent:.1f}%) | "
                f"Speed: {fps:,.0f} steps/s | ETA: {eta_str} | Mean Reward: {reward_str}"
            )
            self.last_print_step = current_step

        return True





def parse_level_weights(
    weights_str: Optional[str],
    active_levels: Optional[List[Union[int, str]]] = None,
) -> Optional[Dict[str, float]]:
    """Parse a level weights string (e.g. '1M:3,2M:2,3M:1,4M:1') into a dictionary.

    If active_levels is provided, any level not mentioned in weights_str defaults to 1.0.
    """
    if weights_str is None or not str(weights_str).strip():
        return None

    weights: Dict[str, float] = {}
    tokens = [tok.strip() for tok in str(weights_str).split(",") if tok.strip()]
    for token in tokens:
        if ":" not in token:
            raise ValueError(
                f"Invalid level weight format '{token}'. Expected 'LEVEL:WEIGHT' (e.g. '1M:2')."
            )
        parts = token.split(":", 1)
        lvl = normalize_level_tag(parts[0].strip())
        try:
            w = float(parts[1].strip())
        except ValueError:
            raise ValueError(f"Invalid weight '{parts[1]}' for level '{lvl}'. Must be a number.")
        if w <= 0:
            raise ValueError(f"Weight for level '{lvl}' must be strictly positive, got {w}")
        weights[lvl] = w

    if active_levels is not None:
        normalized_active = [normalize_level_tag(lvl) for lvl in active_levels]
        for lvl in normalized_active:
            if lvl not in weights:
                weights[lvl] = 1.0

    return weights


def compute_piece_sampling_weights(
    train_items: List[Dict[str, Any]],
    level_weights: Dict[str, float],
) -> List[float]:
    """Compute per-piece sampling weights so total probability per level matches level_weights.

    If level L has N_L pieces and weight W_L, each piece in L gets weight W_L / N_L.
    """
    counts: Dict[str, int] = {}
    for it in train_items:
        lvl = normalize_level_tag(it["level"])
        counts[lvl] = counts.get(lvl, 0) + 1

    weights: List[float] = []
    for it in train_items:
        lvl = normalize_level_tag(it["level"])
        w_level = level_weights.get(lvl, 1.0)
        n_level = counts.get(lvl, 1)
        weights.append(w_level / n_level)

    return weights


def make_env(
    scores: List[Score],
    seed: int,
    rank: int,
    reward_config: Optional[RewardConfig] = None,
    score_weights: Optional[List[float]] = None,
):
    """Factory function for vectorized environments."""
    def _init():
        torch.set_num_threads(1)
        env = MultiKeyPianoEnv(
            scores=scores,
            seed=seed + rank * 1000,
            reward_config=reward_config,
            score_weights=score_weights,
        )
        env = Monitor(env)
        return env

    return _init


def load_multikey_training_pieces(
    manifest_path: Path,
    levels: List[Union[int, str]],
    data_dir: Path = Path("data"),
    return_items: bool = False,
) -> Union[List[Score], tuple[List[Score], List[Dict[str, Any]]]]:
    """Read manifest_multikey.json and load ONLY train-split pieces for the chosen levels."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at '{manifest_path}'. Run scripts/generate_multikey_dataset.py first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    target_levels = {normalize_level_tag(lvl) for lvl in levels}

    # Filter strictly for train split and matching levels
    train_items = [
        item
        for item in manifest
        if item["split"] == "train" and normalize_level_tag(item["level"]) in target_levels
    ]

    # Security check: Ensure no held-out piece can ever leak into the training pool
    for item in train_items:
        if item["split"] != "train":
            raise ValueError(f"Security check failed: Non-train item detected: {item}")

    print(f"Loaded {len(train_items)} training pieces across levels {sorted(target_levels)}:")
    for lvl in sorted(target_levels):
        count = sum(1 for it in train_items if normalize_level_tag(it["level"]) == lvl)
        print(f"  Level {lvl}: {count} pieces")

    scores = [load_score(data_dir / item["filename"]) for item in train_items]
    if return_items:
        return scores, train_items
    return scores


def train_multikey(
    levels: List[Union[int, str]] = ["1M"],
    timesteps: int = 300000,
    n_envs: int = 4,
    seed: int = 0,
    run_name: str = "multikey_level1_run",
    resume_from: Optional[str] = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    ent_coef: float = 0.001,
    exact_reward: float = 1.0,
    off_by_one_reward: float = 0.5,
    wrong_press_penalty: float = -0.5,
    miss_penalty: float = -1.0,
    neighbor_key_penalty: float = 0.0,
    repeat_penalty: float = 0.0,
    policy: str = "pitch_conv",
    level_weights: Optional[Union[str, Dict[str, float]]] = None,
    scores_override: Optional[List[Score]] = None,
    score_weights_override: Optional[List[float]] = None,
) -> Path:
    """Train a multi-key PPO model and save final checkpoint."""
    data_dir = Path("data")
    manifest_path = data_dir / "manifest_multikey.json"

    target_levels = [normalize_level_tag(lvl) for lvl in levels]
    parsed_weights = None
    if isinstance(level_weights, str):
        parsed_weights = parse_level_weights(level_weights, target_levels)
    elif isinstance(level_weights, dict):
        parsed_weights = {normalize_level_tag(k): float(v) for k, v in level_weights.items()}
        for lvl in target_levels:
            if lvl not in parsed_weights:
                parsed_weights[lvl] = 1.0

    score_weights: Optional[List[float]] = None
    if scores_override is not None:
        scores = scores_override
        score_weights = score_weights_override
    else:
        scores, train_items = load_multikey_training_pieces(
            manifest_path, levels, data_dir, return_items=True
        )
        if parsed_weights is not None:
            score_weights = compute_piece_sampling_weights(train_items, parsed_weights)
            tot_w = sum(parsed_weights.values())
            print("\nPer-level sampling weights active:")
            for lvl in sorted(parsed_weights.keys()):
                pct = (parsed_weights[lvl] / tot_w) * 100 if tot_w > 0 else 0
                print(f"  Level {lvl}: weight {parsed_weights[lvl]:.2f} (~{pct:.1f}% chance per episode)")

    reward_config = RewardConfig(
        hit_exact=exact_reward,
        hit_off_by_one=off_by_one_reward,
        wrong_press=wrong_press_penalty,
        miss=miss_penalty,
        neighbor_key_penalty=neighbor_key_penalty,
        repeat_penalty=repeat_penalty,
    )

    print("\nReward configuration:")
    print(f"  Exact hit reward:       {reward_config.hit_exact:+.2f}")
    print(f"  Off-by-one hit reward:  {reward_config.hit_off_by_one:+.2f}")
    print(f"  Wrong press penalty:    {reward_config.wrong_press:+.2f}")
    print(f"  Miss penalty:           {reward_config.miss:+.2f}")
    if reward_config.neighbor_key_penalty != 0.0:
        print(f"  Neighbor key penalty:   {-abs(reward_config.neighbor_key_penalty):+.2f} (extra)")
    if reward_config.repeat_penalty != 0.0:
        print(f"  Repeat penalty:         {-abs(reward_config.repeat_penalty):+.2f} (extra)")

    print(f"\nSetting up {n_envs} parallel environment(s) (CPU only, 1 thread/worker)...")
    if n_envs > 1:
        env = SubprocVecEnv(
            [
                make_env(scores, seed, i, reward_config=reward_config, score_weights=score_weights)
                for i in range(n_envs)
            ]
        )
    else:
        env = DummyVecEnv(
            [make_env(scores, seed, 0, reward_config=reward_config, score_weights=score_weights)]
        )

    # Output paths
    checkpoints_dir = Path("checkpoints") / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path("runs") / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Save reward and run config
    reward_config_path = checkpoints_dir / "reward_config.json"
    with open(reward_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "hit_exact": reward_config.hit_exact,
                "hit_off_by_one": reward_config.hit_off_by_one,
                "wrong_press": reward_config.wrong_press,
                "miss": reward_config.miss,
                "neighbor_key_penalty": reward_config.neighbor_key_penalty,
                "repeat_penalty": reward_config.repeat_penalty,
            },
            f,
            indent=2,
        )

    run_config_path = checkpoints_dir / "run_config.json"
    normalized_lvls = [normalize_level_tag(lvl) for lvl in levels]
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": policy,
                "levels": normalized_lvls,
                "level_weights": parsed_weights,
                "timesteps": timesteps,
                "learning_rate": learning_rate,
                "ent_coef": ent_coef,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "seed": seed,
                "n_envs": n_envs,
            },
            f,
            indent=2,
        )

    # Policy selection
    if policy == "pitch_conv":
        policy_class = MultiKeyPitchConvPolicy
        policy_kwargs: Dict[str, Any] = {}
        print("Using policy: MultiKeyPitchConvPolicy (1D shared pitch convolution + Bernoulli head)")
    elif policy == "mlp":
        policy_class = "MlpPolicy"
        policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
        print("Using policy: MlpPolicy (standard MLP: [256, 256])")
    else:
        raise ValueError(f"Unknown policy '{policy}'. Must be 'pitch_conv' or 'mlp'.")

    # Resume or create new model
    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Model to resume from not found at '{resume_path}'")
        print(f"Resuming training from checkpoint: {resume_path}")
        custom_objects = {"MultiKeyPitchConvPolicy": MultiKeyPitchConvPolicy}
        model = PPO.load(
            str(resume_path),
            env=env,
            device="cpu",
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            tensorboard_log=str(runs_dir),
            custom_objects=custom_objects,
        )
    else:
        model = PPO(
            policy=policy_class,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            seed=seed,
            device="cpu",
            tensorboard_log=str(runs_dir),
            policy_kwargs=policy_kwargs,
            verbose=0,
        )

    # Callbacks
    progress_callback = PlainProgressCallback(
        total_timesteps=timesteps,
        print_freq=max(100, min(5000, timesteps // 10)),
    )
    tb_callback = MultiKeyTensorboardCallback(log_freq=max(32, min(256, n_steps)))
    callbacks: List[BaseCallback] = [progress_callback, tb_callback]

    if timesteps >= 50000:
        checkpoint_freq = max(10000, timesteps // 5)
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_freq // n_envs,
            save_path=str(checkpoints_dir),
            name_prefix="checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        callbacks.append(checkpoint_callback)

    # Train
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=not bool(resume_from),
        )
    finally:
        env.close()

    # Save final model
    final_path = checkpoints_dir / "final.zip"
    model.save(str(final_path))
    print(f"\nTraining complete! Final model saved to: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO multi-key piano agent on procedural polyphony."
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["1M"],
        help="Curriculum levels to train on (e.g. 1M or 1 2 3).",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300000,
        help="Total training steps (default: 300,000).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="multikey_level1_run",
        help="Subdirectory name in checkpoints/ and runs/.",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel CPU worker environments (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to an existing .zip checkpoint to continue training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Adam learning rate (default: 0.0003).",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.001,
        help="Entropy bonus coefficient for MultiBinary(88) (default: 0.001).",
    )
    parser.add_argument(
        "--exact-reward",
        type=float,
        default=1.0,
        help="Reward for striking correct key at exact start step (default: +1.0).",
    )
    parser.add_argument(
        "--off-by-one-reward",
        type=float,
        default=0.5,
        help="Reward for striking correct key 1 step early/late (default: +0.5).",
    )
    parser.add_argument(
        "--wrong-press-penalty",
        type=float,
        default=-0.5,
        help="Penalty for striking an unmatched key (default: -0.5).",
    )
    parser.add_argument(
        "--miss-penalty",
        type=float,
        default=-1.0,
        help="Penalty for missing a scheduled note (default: -1.0).",
    )
    parser.add_argument(
        "--neighbor-key-penalty",
        type=float,
        default=0.0,
        help="Additional penalty for pressing a neighbor key within 1-2 semitones of an active note (default: 0.0).",
    )
    parser.add_argument(
        "--repeat-penalty",
        type=float,
        default=0.0,
        help="Additional penalty for double-striking an already-active or just-struck note within +-2 steps (default: 0.0).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="pitch_conv",
        choices=["pitch_conv", "mlp"],
        help="Policy architecture: 'pitch_conv' (default) or 'mlp'.",
    )
    parser.add_argument(
        "--level-weights",
        type=str,
        default=None,
        help="Optional sampling weights per level (e.g. '1M:3,2M:2,3M:1,4M:1'). Defaults to uniform sampling.",
    )

    args = parser.parse_args()

    train_multikey(
        levels=args.levels,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        run_name=args.run_name,
        resume_from=args.resume_from,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        exact_reward=args.exact_reward,
        off_by_one_reward=args.off_by_one_reward,
        wrong_press_penalty=args.wrong_press_penalty,
        miss_penalty=args.miss_penalty,
        neighbor_key_penalty=args.neighbor_key_penalty,
        repeat_penalty=args.repeat_penalty,
        policy=args.policy,
        level_weights=args.level_weights,
    )


if __name__ == "__main__":
    main()
