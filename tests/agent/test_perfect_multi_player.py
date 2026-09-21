"""Unit tests for PerfectMultiPlayer and multi-key evaluation metrics."""

import numpy as np
import pytest

from pianorl.agent import PerfectMultiPlayer
from pianorl.data import generate_multi_key_score
from pianorl.env.multi_keys_env import MultiKeyPianoEnv
from pianorl.eval.multi_key_eval import (
    compute_array_metrics,
    evaluate_multi_player,
)


def test_perfect_multi_player_achieves_1_f1_across_levels():
    """Verify PerfectMultiPlayer scores 1.0 F1, 100% exact rate, and 0 mistakes on all levels (1M to 8M)."""
    player = PerfectMultiPlayer()

    for level in range(1, 9):
        score = generate_multi_key_score(level, seed=42)
        env = MultiKeyPianoEnv(scores=[score], seed=42)

        obs, info = env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = player.act(obs)
            assert isinstance(action, np.ndarray)
            assert action.shape == (88,)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        # Perfect sight-reading checks
        assert info["hits_exact"] == info["total_notes"], f"Level {level}M: missed exact hits"
        assert info["hits_off_by_one"] == 0, f"Level {level}M: off-by-one strikes occurred"
        assert info["wrong_presses"] == 0, f"Level {level}M: wrong strikes occurred"
        assert info["missed_notes"] == 0, f"Level {level}M: notes missed"
        assert total_reward == pytest.approx(float(info["total_notes"])), (
            f"Level {level}M: total reward does not match exact hit count"
        )


def test_perfect_multi_player_predict_interface():
    """Verify predict interface works for both single and batched observations."""
    score = generate_multi_key_score(1, seed=99)
    env = MultiKeyPianoEnv(scores=[score], seed=99)
    obs, _ = env.reset()

    player = PerfectMultiPlayer()

    # Single observation
    act, state = player.predict(obs, deterministic=True)
    assert act.shape == (88,)
    assert state is None

    # Batched observation
    batched_obs = np.stack([obs, obs])
    batched_acts, _ = player.predict(batched_obs)
    assert batched_acts.shape == (2, 88)
    np.testing.assert_array_equal(batched_acts[0], batched_acts[1])


def test_compute_array_metrics():
    """Verify binary array evaluation metrics across perfect, wrong, and missed strikes."""
    # 1. Perfect match (3 notes due, 3 notes struck)
    presses = np.zeros(88, dtype=np.int8)
    targets = np.zeros(88, dtype=np.int8)
    presses[[10, 20, 30]] = 1
    targets[[10, 20, 30]] = 1

    m = compute_array_metrics(presses, targets)
    assert m.true_positives == 3
    assert m.false_positives == 0
    assert m.false_negatives == 0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.exact_rate == 1.0
    assert m.chord_exact_rate == 1.0

    # 2. Extra press (speculative spam)
    presses_extra = np.zeros(88, dtype=np.int8)
    presses_extra[[10, 20, 30, 40]] = 1  # key 40 is extra
    m2 = compute_array_metrics(presses_extra, targets)
    assert m2.true_positives == 3
    assert m2.false_positives == 1
    assert m2.precision == 0.75
    assert m2.recall == 1.0
    assert m2.f1 == pytest.approx(2 * 0.75 * 1.0 / 1.75)
    assert m2.chord_exact_rate == 0.0

    # 3. Missed note
    presses_miss = np.zeros(88, dtype=np.int8)
    presses_miss[[10, 20]] = 1  # key 30 missed
    m3 = compute_array_metrics(presses_miss, targets)
    assert m3.true_positives == 2
    assert m3.false_positives == 0
    assert m3.false_negatives == 1
    assert m3.precision == 1.0
    assert m3.recall == pytest.approx(2 / 3)
    assert m3.exact_rate == pytest.approx(2 / 3)


def test_evaluate_multi_player_runner():
    """Verify evaluate_multi_player returns accurate graded metrics across multiple pieces."""
    scores = [
        generate_multi_key_score(1, seed=1),
        generate_multi_key_score(2, seed=2),
    ]
    items = [{"level": "1M"}, {"level": "2M"}]

    player = PerfectMultiPlayer()
    results = evaluate_multi_player(player, scores, split_items=items)

    assert "Level 1M" in results
    assert "Level 2M" in results
    assert "Overall" in results

    assert results["Overall"].precision == 1.0
    assert results["Overall"].recall == 1.0
    assert results["Overall"].f1 == 1.0
    assert results["Overall"].exact_rate == 1.0
    assert results["Overall"].chord_exact_rate == 1.0
