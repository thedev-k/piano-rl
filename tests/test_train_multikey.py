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


def test_parse_level_weights():
    """Verify level weights parsing and defaulting behavior."""
    from scripts.train_multikey import parse_level_weights

    # None or empty returns None
    assert parse_level_weights(None) is None
    assert parse_level_weights("") is None
    assert parse_level_weights("   ") is None

    # Valid string with whitespace
    weights = parse_level_weights("1M: 3 , 2m : 2, 3M:1", active_levels=["1M", "2M", "3M"])
    assert weights == {"1M": 3.0, "2M": 2.0, "3M": 1.0}

    # Missing active level defaults to 1.0
    weights_partial = parse_level_weights("1M:4", active_levels=["1M", "2M", "3M"])
    assert weights_partial == {"1M": 4.0, "2M": 1.0, "3M": 1.0}

    # Error handling
    with pytest.raises(ValueError, match="Invalid level weight format"):
        parse_level_weights("1M")

    with pytest.raises(ValueError, match="strictly positive"):
        parse_level_weights("1M:0")

    with pytest.raises(ValueError, match="strictly positive"):
        parse_level_weights("1M:-2")

    with pytest.raises(ValueError, match="Must be a number"):
        parse_level_weights("1M:abc")


def test_compute_piece_sampling_weights():
    """Verify per-piece sampling weights produce exact level probabilities."""
    from scripts.train_multikey import compute_piece_sampling_weights

    # Level 1M has 2 pieces, Level 2M has 4 pieces
    train_items = [
        {"level": "1M", "filename": "1.mid"},
        {"level": "1M", "filename": "2.mid"},
        {"level": "2M", "filename": "3.mid"},
        {"level": "2M", "filename": "4.mid"},
        {"level": "2M", "filename": "5.mid"},
        {"level": "2M", "filename": "6.mid"},
    ]
    # Weight Level 1M with 2.0 and Level 2M with 1.0
    level_weights = {"1M": 2.0, "2M": 1.0}
    weights = compute_piece_sampling_weights(train_items, level_weights)

    assert len(weights) == 6
    # Level 1M pieces: 2.0 / 2 = 1.0 each
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(1.0)
    # Level 2M pieces: 1.0 / 4 = 0.25 each
    assert weights[2] == pytest.approx(0.25)
    assert weights[3] == pytest.approx(0.25)
    assert weights[4] == pytest.approx(0.25)
    assert weights[5] == pytest.approx(0.25)

    # Sum of Level 1M weights = 2.0; Sum of Level 2M weights = 1.0
    assert sum(weights[:2]) == pytest.approx(2.0)
    assert sum(weights[2:]) == pytest.approx(1.0)


def test_multikey_env_weighted_sampling():
    """Verify MultiKeyPianoEnv samples pieces according to score_weights."""
    from pianorl.env import MultiKeyPianoEnv

    s1 = generate_multi_key_score(1, seed=1)
    s2 = generate_multi_key_score(2, seed=2)

    # Validation errors
    with pytest.raises(ValueError, match="Length of score_weights"):
        MultiKeyPianoEnv(scores=[s1, s2], score_weights=[1.0])

    with pytest.raises(ValueError, match="strictly positive"):
        MultiKeyPianoEnv(scores=[s1, s2], score_weights=[1.0, 0.0])

    # Empirical test: score 0 has 80% weight, score 1 has 20% weight
    env = MultiKeyPianoEnv(scores=[s1, s2], score_weights=[8.0, 2.0], seed=42)
    counts = {0: 0, 1: 0}
    for _ in range(500):
        env.reset()
        if env.current_score is s1:
            counts[0] += 1
        elif env.current_score is s2:
            counts[1] += 1

    prob_s1 = counts[0] / 500
    # Expected: ~0.80. Allow reasonable statistical tolerance [0.72, 0.88]
    assert 0.72 <= prob_s1 <= 0.88, f"Observed frequency {prob_s1:.2f} deviated too much from 0.80"

