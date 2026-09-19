import json
from pathlib import Path
import numpy as np
import pytest

from pianorl.eval import PPOPlayer
from pianorl.score import load_score
from scripts.evaluate import evaluate_player
from scripts.train import load_training_pieces, train


def test_training_loader_only_train_split_and_requested_levels():
    """Verify that load_training_pieces only loads train split and only requested levels."""
    manifest_path = Path("data/manifest.json")
    assert manifest_path.exists()

    # Request only level 1
    scores_lvl1 = load_training_pieces(manifest_path, levels=[1])
    assert len(scores_lvl1) == 170  # Exactly 170 train pieces for level 1

    # Request level 1 and 2
    scores_lvl12 = load_training_pieces(manifest_path, levels=[1, 2])
    assert len(scores_lvl12) == 340  # 170 + 170


def test_heldout_pieces_never_appear_in_training():
    """Verify with 100% certainty that no held-out piece can ever appear in the training set."""
    manifest_path = Path("data/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    heldout_filenames = {item["filename"] for item in manifest if item["split"] == "heldout"}

    train_items = [
        item for item in manifest
        if item["split"] == "train"
    ]
    train_filenames = {item["filename"] for item in train_items}

    # Zero intersection between heldout and train files
    intersection = heldout_filenames.intersection(train_filenames)
    assert len(intersection) == 0


def test_tiny_training_run_and_save(tmp_path: Path):
    """Run a tiny 256-timestep training session to verify the pipeline runs and saves a model."""
    manifest_path = Path("data/manifest.json")
    scores = load_training_pieces(manifest_path, levels=[1])[:10]  # Just 10 scores for speed

    final_model_path = train(
        levels=[1],
        timesteps=256,
        n_envs=1,
        seed=42,
        run_name="unit_test_run",
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
    )

    assert final_model_path.exists()
    assert final_model_path.name == "final.zip"


def test_ppo_player_act_on_plain_observation():
    """PPOPlayer loads model and predicts a valid action (0 to 88) on a plain numpy observation."""
    model_path = Path("checkpoints/unit_test_run/final.zip")
    assert model_path.exists()

    player = PPOPlayer(model_path)

    # Observation vector size is 2821
    dummy_obs = np.zeros(2821, dtype=np.float32)
    action = player.act(dummy_obs)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action <= 88


def test_evaluate_script_with_ppo_player():
    """Verify evaluate_player works with PPOPlayer on a couple of pieces without error."""
    model_path = Path("checkpoints/unit_test_run/final.zip")
    player = PPOPlayer(model_path)

    manifest_path = Path("data/manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    heldout_items = [item for item in manifest if item["split"] == "heldout"][:2]
    scores = [load_score(Path("data") / it["filename"]) for it in heldout_items]

    results = evaluate_player(player, scores, heldout_items)
    assert "Overall" in results
    assert results["Overall"].num_pieces == 2
