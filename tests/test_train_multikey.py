"""Smoke tests and integration tests for multi-key PPO training pipeline."""

import json
from pathlib import Path
import numpy as np
import pytest
from stable_baselines3 import PPO

from pianorl.agent import MultiKeyPitchConvPolicy
from pianorl.data import generate_multi_key_score
from scripts.train_multikey import (
    load_multikey_training_pieces,
    normalize_level_tag,
    train_multikey,
)


def test_normalize_level_tag():
    """Verify level tag standardization to '1M', '2M', etc."""
    assert normalize_level_tag(1) == "1M"
    assert normalize_level_tag("1") == "1M"
    assert normalize_level_tag("1m") == "1M"
    assert normalize_level_tag("1M") == "1M"
    assert normalize_level_tag("8M") == "8M"


def test_multikey_training_loader_only_train_split():
    """Verify load_multikey_training_pieces loads only train pieces for requested levels."""
    manifest_path = Path("data/manifest_multikey.json")
    assert manifest_path.exists(), "Manifest must exist on disk"

    # Request Level 1M
    scores_1m = load_multikey_training_pieces(manifest_path, levels=["1M"])
    assert len(scores_1m) == 170  # Exactly 170 train pieces

    # Request Level 1M and 2M
    scores_12m = load_multikey_training_pieces(manifest_path, levels=[1, 2])
    assert len(scores_12m) == 340  # 170 + 170


def test_heldout_multikey_never_in_training():
    """Verify zero overlap between heldout and train files in multikey manifest."""
    manifest_path = Path("data/manifest_multikey.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    heldout_files = {it["filename"] for it in manifest if it["split"] == "heldout"}
    train_files = {it["filename"] for it in manifest if it["split"] == "train"}

    assert len(heldout_files.intersection(train_files)) == 0


def test_multikey_smoke_train_step(tmp_path: Path):
    """Smoke test: run train_multikey for just 64 steps to prove env, policy, and PPO communicate."""
    # Use 2 synthetic scores for blazing fast test execution
    toy_scores = [
        generate_multi_key_score(1, seed=10),
        generate_multi_key_score(2, seed=20),
    ]

    final_model_path = train_multikey(
        levels=["1M"],
        timesteps=64,
        n_envs=1,
        seed=42,
        run_name="smoke_test_multikey",
        n_steps=64,
        batch_size=32,
        scores_override=toy_scores,
    )

    assert final_model_path.exists()
    assert final_model_path.name == "final.zip"

    # Verify model loads and predicts valid MultiBinary(88) action arrays
    custom_objects = {"MultiKeyPitchConvPolicy": MultiKeyPitchConvPolicy}
    model = PPO.load(str(final_model_path), custom_objects=custom_objects)
    dummy_obs = np.zeros(2821, dtype=np.float32)

    action, _ = model.predict(dummy_obs, deterministic=True)
    assert isinstance(action, np.ndarray)
    assert action.shape == (88,)
    assert set(np.unique(action)).issubset({0, 1})
