import argparse
import json
from pathlib import Path
import pytest

from pianorl.env import PianoFreeKeysEnv, RewardConfig
from pianorl.eval.diagnosis import categorize_wrong_press, DiagnosisStats
from scripts.train import make_env, train
from pianorl.score import Score, NoteEvent


def test_categorize_wrong_press_repeat():
    """Verify category 'repeat': pressed pitch matches a note within +-2 steps."""
    # Note at step 10 with pitch 60
    targets = [{"pitch": 60, "start_step": 10, "hit": False}]

    # Same pitch at steps 8, 9, 10, 11, 12 should all be 'repeat'
    for step in [8, 9, 10, 11, 12]:
        cat = categorize_wrong_press(pressed_pitch=60, current_step=step, targets=targets)
        assert cat == "repeat", f"Expected repeat at step {step}, got {cat}"


def test_categorize_wrong_press_wrong_key_near():
    """Verify category 'wrong_key_near': a note is within +-2 steps, but wrong pitch."""
    # Note at step 10 with pitch 60
    targets = [{"pitch": 60, "start_step": 10, "hit": False}]

    # Different pitch (62, D4) within +-2 steps (steps 8, 9, 10, 11, 12)
    for step in [8, 9, 10, 11, 12]:
        cat = categorize_wrong_press(pressed_pitch=62, current_step=step, targets=targets)
        assert cat == "wrong_key_near", f"Expected wrong_key_near at step {step}, got {cat}"


def test_categorize_wrong_press_no_note_nearby():
    """Verify category 'no_note_nearby': no note starts within +-2 steps."""
    # Note at step 10 with pitch 60
    targets = [{"pitch": 60, "start_step": 10, "hit": False}]

    # Steps 7 and 13 are 3 steps away (> 2 steps), so no note is nearby
    for step in [0, 5, 7, 13, 20]:
        cat = categorize_wrong_press(pressed_pitch=60, current_step=step, targets=targets)
        assert cat == "no_note_nearby", f"Expected no_note_nearby at step {step}, got {cat}"

        cat_other = categorize_wrong_press(pressed_pitch=65, current_step=step, targets=targets)
        assert cat_other == "no_note_nearby", f"Expected no_note_nearby at step {step}, got {cat_other}"


def test_diagnosis_stats_calculations():
    """Verify DiagnosisStats calculates percentages, recall, precision, and presses_per_note."""
    stats = DiagnosisStats(
        pieces_count=1,
        total_notes=10,
        total_presses=15,
        hits_exact=6,
        hits_off_by_one=2,
        missed_notes=2,
        wrong_repeat=3,
        wrong_key_near=3,
        wrong_no_note=1,
    )

    assert stats.total_wrong == 7
    assert pytest.approx(stats.repeat_pct, 0.1) == 42.857
    assert pytest.approx(stats.wrong_key_near_pct, 0.1) == 42.857
    assert pytest.approx(stats.no_note_nearby_pct, 0.1) == 14.286
    assert pytest.approx(stats.presses_per_note, 0.01) == 1.5
    # TP = 8, attempts = 8 + 7 = 15 -> precision = 8/15
    assert pytest.approx(stats.precision, 0.01) == 8 / 15
    # TP = 8, total_target = 8 + 2 = 10 -> recall = 8/10 = 0.8
    assert pytest.approx(stats.recall, 0.01) == 0.8


def test_train_make_env_reward_config():
    """Verify that make_env propagates custom and default RewardConfig properly."""
    dummy_score = Score(notes=[NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0)], tempo_bpm=60.0)

    # 1. Default RewardConfig
    fn_default = make_env(scores=[dummy_score], seed=0, rank=0)
    env_default = fn_default()
    assert env_default.unwrapped.reward_config.hit_exact == 1.0
    assert env_default.unwrapped.reward_config.hit_off_by_one == 0.5
    assert env_default.unwrapped.reward_config.wrong_press == -0.5
    assert env_default.unwrapped.reward_config.miss == -1.0
    env_default.close()

    # 2. Custom RewardConfig
    custom_rc = RewardConfig(
        hit_exact=2.5,
        hit_off_by_one=1.0,
        wrong_press=-1.2,
        miss=-2.0,
    )
    fn_custom = make_env(scores=[dummy_score], seed=0, rank=0, reward_config=custom_rc)
    env_custom = fn_custom()
    assert env_custom.unwrapped.reward_config.hit_exact == 2.5
    assert env_custom.unwrapped.reward_config.hit_off_by_one == 1.0
    assert env_custom.unwrapped.reward_config.wrong_press == -1.2
    assert env_custom.unwrapped.reward_config.miss == -2.0
    env_custom.close()


def test_train_saves_reward_config_json(tmp_path: Path):
    """Verify that train() saves reward_config.json with specified reward parameters."""
    run_name = "test_reward_config_run"
    checkpoints_dir = Path("checkpoints") / run_name

    final_path = train(
        levels=[1],
        timesteps=128,
        n_envs=1,
        seed=123,
        run_name=run_name,
        n_steps=128,
        batch_size=64,
        exact_reward=1.8,
        off_by_one_reward=0.8,
        wrong_press_penalty=-0.9,
        miss_penalty=-1.5,
    )

    reward_json_path = checkpoints_dir / "reward_config.json"
    assert reward_json_path.exists()

    with open(reward_json_path, "r", encoding="utf-8") as f:
        saved_rewards = json.load(f)

    assert saved_rewards["hit_exact"] == 1.8
    assert saved_rewards["hit_off_by_one"] == 0.8
    assert saved_rewards["wrong_press"] == -0.9
    assert saved_rewards["miss"] == -1.5
