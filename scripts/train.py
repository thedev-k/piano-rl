import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from pianorl.env import PianoFreeKeysEnv
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

            # Recent episode rewards from monitor
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


def make_env(scores: List[Score], seed: int, rank: int):
    def _init():
        torch.set_num_threads(1)
        env = PianoFreeKeysEnv(scores=scores, seed=seed + rank * 1000)
        env = Monitor(env)
        return env

    return _init


def load_training_pieces(
    manifest_path: Path,
    levels: List[int],
    data_dir: Path = Path("data"),
) -> List[Score]:
    """Read manifest.json and load ONLY train-split pieces for the chosen levels."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at '{manifest_path}'. Run generate_dataset.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Filter strictly for train split and matching levels
    train_items = [
        item for item in manifest
        if item["split"] == "train" and item["level"] in levels
    ]

    # Verify that zero heldout pieces are included
    for item in train_items:
        if item["split"] != "train":
            raise ValueError(f"Security check failed: Non-train item detected: {item}")

    print(f"Loaded {len(train_items)} training pieces across levels {levels}:")
    for lvl in sorted(set(levels)):
        count = sum(1 for it in train_items if it["level"] == lvl)
        print(f"  Level {lvl}: {count} pieces")

    scores = [load_score(data_dir / item["filename"]) for item in train_items]
    return scores


def train(
    levels: List[int] = [1],
    timesteps: int = 300000,
    n_envs: int = 4,
    seed: int = 0,
    run_name: str = "level1_run",
    resume_from: str = None,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    ent_coef: float = 0.01,
) -> Path:
    data_dir = Path("data")
    manifest_path = data_dir / "manifest.json"
    scores = load_training_pieces(manifest_path, levels, data_dir)

    print(f"\nSetting up {n_envs} parallel environment(s) (CPU only, 1 thread/worker)...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env(scores, seed, i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(scores, seed, 0)])

    # Output paths
    checkpoints_dir = Path("checkpoints") / run_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path("runs") / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Model architecture: 2 layers of 256 units
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if resume_from:
        resume_path = Path(resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Model to resume from not found at '{resume_path}'")
        print(f"Resuming training from existing checkpoint: {resume_path}")
        model = PPO.load(
            str(resume_path),
            env=env,
            device="cpu",
            tensorboard_log=str(runs_dir),
            learning_rate=learning_rate,
            ent_coef=ent_coef,
        )
    else:
        print("Creating new PPO agent (MLP: [256, 256], ent_coef=0.01)...")
        model = PPO(
            "MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            tensorboard_log=str(runs_dir),
            seed=seed,
            device="cpu",
            verbose=0,
        )

    # Callbacks
    # Save checkpoint every 50,000 steps
    save_freq = max(1, 50000 // n_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(checkpoints_dir),
        name_prefix="checkpoint",
    )

    # Progress printer: print every 10,000 steps or every 2,000 if short run
    print_freq = 2000 if timesteps < 20000 else 10000
    progress_callback = PlainProgressCallback(
        total_timesteps=timesteps,
        print_freq=print_freq,
    )

    start_wall = time.time()
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=False,
        )
    finally:
        env.close()

    total_time = time.time() - start_wall
    final_path = checkpoints_dir / "final.zip"
    model.save(str(final_path))

    overall_fps = timesteps / total_time if total_time > 0 else 0
    print("\n" + "=" * 65)
    print("Training finished successfully!")
    print(f"  Total time:     {int(total_time // 60)}m {int(total_time % 60):02d}s")
    print(f"  Average speed:  {overall_fps:,.0f} timesteps/second")
    print(f"  Saved model:    {final_path}")
    print(f"  TensorBoard:    runs/{run_name}")
    print("=" * 65 + "\n")

    return final_path


def main():
    cpu_count = os.cpu_count() or 4
    default_envs = max(1, min(8, cpu_count - 1))

    parser = argparse.ArgumentParser(description="Train a PPO piano agent with Stable-Baselines3.")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1],
        help="Difficulty level(s) to train on (e.g. --levels 1 or --levels 1 2)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=300000,
        help="Total training timesteps (default: 300,000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=default_envs,
        help=f"Number of parallel environments (default on this system: {default_envs})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for environments and training (default: 0)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="level1_run",
        help="Name of this run (used for checkpoints and tensorboard logs)",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional path to a saved model zip file to resume training from",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for Adam optimizer (default: 0.0003)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment per update (default: 2048)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size for PPO updates (default: 64)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="Entropy coefficient to encourage exploration (default: 0.01)",
    )

    args = parser.parse_args()

    train(
        levels=args.levels,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        run_name=args.run_name,
        resume_from=args.resume_from,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
    )


if __name__ == "__main__":
    main()
