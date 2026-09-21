import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO

from pianorl.agent import MultiKeyPitchConvPolicy
from pianorl.env import MultiKeyPianoEnv
from pianorl.score import NoteEvent, Score


def make_test_score():
    return Score(
        notes=[NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0)],
        tempo_bpm=120.0,
    )


def test_multi_key_policy_initialization():
    """Verify MultiKeyPitchConvPolicy initializes with MultiBinary(88) and bias = -3.0."""
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.MultiBinary(88)

    policy = MultiKeyPitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    # 1. Action net bias initialization trick: must be exactly -3.0
    assert policy.action_net.bias.item() == pytest.approx(-3.0)

    # 2. Action net layer type: Linear(64, 1) shared across all keys
    assert policy.action_net.in_features == 64
    assert policy.action_net.out_features == 1

    # 3. Action distribution: Bernoulli distribution over 88 keys
    assert policy.action_dist.action_dims == 88


def test_multi_key_policy_output_shapes():
    """Verify forward network produces (batch, 88) logits and (batch, 1) values."""
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.MultiBinary(88)

    policy = MultiKeyPitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    batch_size = 4
    dummy_obs = th.zeros((batch_size, 2821), dtype=th.float32)

    with th.no_grad():
        logits, values = policy._forward_network(dummy_obs)
        actions, values_fwd, log_prob = policy.forward(dummy_obs)
        val_eval, log_prob_eval, entropy = policy.evaluate_actions(dummy_obs, actions)

    # 88 independent logits (one per key) and 1 value estimate
    assert logits.shape == (batch_size, 88)
    assert values.shape == (batch_size, 1)
    assert actions.shape == (batch_size, 88)
    assert log_prob.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert th.allclose(values, values_fwd)

    # Actions must be binary (0 or 1)
    unique_vals = set(actions.cpu().numpy().flatten().tolist())
    assert unique_vals.issubset({0.0, 1.0})


def test_initial_action_probabilities_around_4_7_percent():
    """Verify initial forward pass yields key-strike probabilities tightly around 4.7%."""
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.MultiBinary(88)

    policy = MultiKeyPitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    dummy_obs = th.randn((8, 2821), dtype=th.float32)

    with th.no_grad():
        logits, _ = policy._forward_network(dummy_obs)
        probs = th.sigmoid(logits)

    mean_prob = probs.mean().item()
    min_prob = probs.min().item()
    max_prob = probs.max().item()

    # sigmoid(-3.0) = 0.04742587... (~4.74%)
    expected_prob = 1.0 / (1.0 + np.exp(3.0))

    assert mean_prob == pytest.approx(expected_prob, abs=0.005), (
        f"Mean probability {mean_prob:.4f} differs from expected ~{expected_prob:.4f}"
    )
    # Ensure no keys have extreme outlier probabilities initially
    assert 0.03 <= min_prob <= 0.06
    assert 0.03 <= max_prob <= 0.06


def test_deterministic_action_is_rest_initially():
    """Verify that in deterministic mode, the initial policy outputs rest (zeros) for all keys."""
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.MultiBinary(88)

    policy = MultiKeyPitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    dummy_obs = th.zeros((2, 2821), dtype=th.float32)

    with th.no_grad():
        actions_det, _, _ = policy.forward(dummy_obs, deterministic=True)

    # In deterministic mode, action = (logit > 0.0). Since logits are ~ -3.0, all actions should be 0
    assert th.all(actions_det == 0.0), "Initial deterministic action should be rest (all zeros)"


def test_shift_equivariance_multi_key():
    """Verify that shifting a note by +3 keys shifts the multi-binary action logits by exactly +3 positions."""
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.MultiBinary(88)

    policy = MultiKeyPitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    k = 25
    k_shifted = k + 3

    # Obs 1: note onset at key k, slot 0
    obs1 = th.zeros((1, 2821), dtype=th.float32)
    obs1[0, k * 16] = 1.0
    obs1[0, 2816] = 1.0  # Beat position one-hot
    obs1[0, 2820] = 0.5  # Tempo

    # Obs 2: note onset shifted to key k+3, slot 0
    obs2 = th.zeros((1, 2821), dtype=th.float32)
    obs2[0, k_shifted * 16] = 1.0
    obs2[0, 2816] = 1.0
    obs2[0, 2820] = 0.5

    with th.no_grad():
        logits1, _ = policy._forward_network(obs1)
        logits2, _ = policy._forward_network(obs2)

    window1 = logits1[0, k - 2 : k + 3]
    window2 = logits2[0, k_shifted - 2 : k_shifted + 3]

    max_diff = (window1 - window2).abs().max().item()
    assert max_diff < 1e-4, f"Shift equivariance failed: max difference {max_diff}"


def test_ppo_multi_key_compatibility():
    """Verify Stable-Baselines3 PPO instantiates cleanly with MultiKeyPitchConvPolicy and MultiKeyPianoEnv."""
    score = make_test_score()
    env = MultiKeyPianoEnv(scores=[score])

    model = PPO(
        MultiKeyPitchConvPolicy,
        env=env,
        device="cpu",
        n_steps=64,
        batch_size=32,
    )

    assert isinstance(model.policy, MultiKeyPitchConvPolicy)
    assert model.action_space.shape == (88,)

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    assert action.shape == (88,)
    assert set(action.tolist()).issubset({0, 1})
