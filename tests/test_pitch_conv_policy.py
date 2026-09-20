import json
from pathlib import Path
import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from pianorl.agent import PitchConvPolicy
from pianorl.env import PianoFreeKeysEnv
from pianorl.eval.players import PPOPlayer
from pianorl.score import Score, NoteEvent
from scripts.train import train


def test_pitch_conv_policy_output_shapes():
    """Verify PitchConvPolicy outputs 89 action logits and 1 value estimate per observation."""
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.Discrete(89)

    policy = PitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    batch_size = 4
    dummy_obs = th.zeros((batch_size, 2821), dtype=th.float32)

    with th.no_grad():
        logits, values = policy._forward_network(dummy_obs)
        actions, values_fwd, log_prob = policy.forward(dummy_obs)

    # 89 logits (1 no-op + 88 keys) and 1 value per observation
    assert logits.shape == (batch_size, 89)
    assert values.shape == (batch_size, 1)
    assert actions.shape == (batch_size,)
    assert log_prob.shape == (batch_size,)
    assert th.allclose(values, values_fwd)


def test_shift_equivariance():
    """SHIFT TEST: verify pitch convolution equivariance and shared weight re-use.

    If a single note onset is shifted from key k to key k+3 (keeping at least 10 keys
    away from both ends of the piano keyboard), the per-key logits must shift by exactly
    3 positions, and the 'press nothing' logit must remain identical.
    """
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(2821,), dtype=np.float32)
    action_space = spaces.Discrete(89)

    policy = PitchConvPolicy(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lambda _: 3e-4,
    )

    k = 30  # Key index 30 (out of 88, safely > 10 away from 0 and 87)
    k_shifted = k + 3  # Key index 33

    # Construct obs1: note onset at key k, slot 0
    obs1 = th.zeros((1, 2821), dtype=th.float32)
    obs1[0, k * 16] = 1.0  # Channel 0, key k, slot 0
    obs1[0, 2816] = 1.0  # Beat position one-hot (slot 0)
    obs1[0, 2820] = 0.5  # Tempo

    # Construct obs2: identical, except note onset is shifted to key k+3, slot 0
    obs2 = th.zeros((1, 2821), dtype=th.float32)
    obs2[0, k_shifted * 16] = 1.0  # Channel 0, key k+3, slot 0
    obs2[0, 2816] = 1.0
    obs2[0, 2820] = 0.5

    with th.no_grad():
        logits1, val1 = policy._forward_network(obs1)
        logits2, val2 = policy._forward_network(obs2)

    # 1. 'Press nothing' logit (index 0) must remain identical
    noop_diff = abs(logits1[0, 0].item() - logits2[0, 0].item())
    assert noop_diff < 1e-4, f"Expected identical no-op logit, got difference {noop_diff}"

    # 2. Value function must remain identical
    val_diff = abs(val1[0, 0].item() - val2[0, 0].item())
    assert val_diff < 1e-4, f"Expected identical value prediction, got difference {val_diff}"

    # 3. Per-key logits must shift by exactly 3 positions
    # Key logits are indices 1..88
    key_logits1 = logits1[0, 1:]  # (88,)
    key_logits2 = logits2[0, 1:]  # (88,)

    # Slice a local window of +-3 keys around the target
    window_original = key_logits1[k - 3 : k + 4]
    window_shifted = key_logits2[k_shifted - 3 : k_shifted + 4]

    shift_diff = (window_original - window_shifted).abs().max().item()
    assert shift_diff < 1e-4, f"Expected shifted key logits to match, max difference: {shift_diff}"


def test_free_keys_env_check_env_with_pitch_conv():
    """Verify Gymnasium check_env passes on PianoFreeKeysEnv."""
    dummy_score = Score(
        notes=[NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0)],
        tempo_bpm=60.0,
    )
    env = PianoFreeKeysEnv(scores=[dummy_score], seed=42)
    check_env(env)
    env.close()


def test_tiny_pitch_conv_training_and_saving():
    """Verify a tiny 512-timestep pitch_conv training run completes and saves artifacts."""
    run_name = "unit_test_pitch_conv_run"
    checkpoints_dir = Path("checkpoints") / run_name

    final_path = train(
        levels=[1],
        timesteps=512,
        n_envs=1,
        seed=42,
        run_name=run_name,
        policy="pitch_conv",
        n_steps=256,
        batch_size=64,
    )

    assert final_path.exists()
    assert final_path.name == "final.zip"

    # Verify run_config.json
    run_config_path = checkpoints_dir / "run_config.json"
    assert run_config_path.exists()
    with open(run_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["policy"] == "pitch_conv"


def test_ppo_player_loads_pitch_conv_and_acts():
    """Verify PPOPlayer seamlessly loads a saved PitchConvPolicy model and acts."""
    model_path = Path("checkpoints/unit_test_pitch_conv_run/final.zip")
    assert model_path.exists()

    player = PPOPlayer(model_path)
    dummy_obs = np.zeros(2821, dtype=np.float32)
    action = player.act(dummy_obs)

    assert isinstance(action, (int, np.integer))
    assert 0 <= action <= 88


def test_mlp_policy_path_still_works():
    """Verify that the standard mlp policy path still functions and saves properly."""
    run_name = "unit_test_mlp_compat_run"
    checkpoints_dir = Path("checkpoints") / run_name

    final_path = train(
        levels=[1],
        timesteps=256,
        n_envs=1,
        seed=42,
        run_name=run_name,
        policy="mlp",
        n_steps=128,
        batch_size=64,
    )

    assert final_path.exists()

    # Verify run_config.json
    run_config_path = checkpoints_dir / "run_config.json"
    assert run_config_path.exists()
    with open(run_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    assert config["policy"] == "mlp"

    player = PPOPlayer(final_path)
    action = player.act(np.zeros(2821, dtype=np.float32))
    assert 0 <= action <= 88


def test_resume_policy_mismatch_raises_error():
    """Verify that resuming an mlp checkpoint with --policy pitch_conv raises a clear error."""
    mlp_model = Path("checkpoints/unit_test_mlp_compat_run/final.zip")
    assert mlp_model.exists()

    with pytest.raises(ValueError, match="Cannot resume training"):
        train(
            levels=[1],
            timesteps=128,
            n_envs=1,
            seed=42,
            run_name="unit_test_mismatch_run",
            resume_from=str(mlp_model),
            policy="pitch_conv",
            n_steps=128,
            batch_size=64,
        )
