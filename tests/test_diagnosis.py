import argparse
import json
from pathlib import Path
import pytest

from pianorl.env import PianoFreeKeysEnv, RewardConfig, split_observation
from pianorl.eval.diagnosis import (
    categorize_wrong_press,
    DiagnosisStats,
    get_semitone_distance,
    get_window_slot_category,
)
from scripts.train import make_env, train
from pianorl.score import Score, NoteEvent
import numpy as np


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


def test_get_window_slot_category():
    """Verify that get_window_slot_category accurately identifies the onset slot and group."""
    # Action 40 corresponds to row 39 (pitch 60, Middle C)
    action = 40
    row = action - 1

    # Total obs length = 2 * 88 * 16 + 4 + 1 = 2821
    obs = np.zeros(2821, dtype=np.float32)

    # 1. Not in window
    slot, cat = get_window_slot_category(obs, action)
    assert slot is None
    assert cat == "not_in_window"

    # 2. Test each slot group
    test_cases = [
        (0, "slots_0_1"),
        (1, "slots_0_1"),
        (2, "slots_2_3"),
        (3, "slots_2_3"),
        (4, "slots_4_7"),
        (6, "slots_4_7"),
        (7, "slots_4_7"),
        (8, "slots_8_15"),
        (15, "slots_8_15"),
    ]

    for target_slot, expected_cat in test_cases:
        window_grid = np.zeros((2, 88, 16), dtype=np.float32)
        window_grid[0, row, target_slot] = 1.0
        obs[: 2 * 88 * 16] = window_grid.flatten()

        slot, cat = get_window_slot_category(obs, action)
        assert slot == target_slot
        assert cat == expected_cat


def test_get_semitone_distance():
    """Verify semitone distance to nearest note start within 2 steps."""
    targets = [
        {"pitch": 60, "start_step": 10, "matched": False},
        {"pitch": 72, "start_step": 11, "matched": False},
    ]

    # At step 10:
    # Pitch 60 -> distance 0 (repeat)
    assert get_semitone_distance(60, 10, targets) == 0

    # Pitch 61 -> distance 1 (61 - 60)
    assert get_semitone_distance(61, 10, targets) == 1

    # Pitch 62 -> distance 2 (62 - 60)
    assert get_semitone_distance(62, 10, targets) == 2

    # Pitch 64 -> distance 4 (nearest is 60, abs(64-60)=4)
    assert get_semitone_distance(64, 10, targets) == 4

    # Pitch 71 -> distance 1 (nearest is 72 at step 11, abs(71-72)=1)
    assert get_semitone_distance(71, 10, targets) == 1

    # At step 20 (> 2 steps from all targets):
    assert get_semitone_distance(60, 20, targets) is None


def test_refined_diagnosis_stats_properties():
    """Verify percentage calculations for refined window and semitone distance properties."""
    stats = DiagnosisStats(
        total_notes=10,
        total_presses=10,
        wrong_repeat=1,
        wrong_key_near=3,
        wrong_no_note=1,
        window_slots_0_1=1,
        window_slots_2_3=1,
        window_slots_4_7=1,
        window_slots_8_15=1,
        window_not_in_window=1,
        dist_0=1,
        dist_1_2=2,
        dist_3_plus=1,
        dist_no_nearby_note=1,
    )
    # Total wrong = 5
    assert stats.total_wrong == 5
    assert pytest.approx(stats.slots_0_1_pct) == 20.0
    assert pytest.approx(stats.slots_2_3_pct) == 20.0
    assert pytest.approx(stats.slots_4_7_pct) == 20.0
    assert pytest.approx(stats.slots_8_15_pct) == 20.0
    assert pytest.approx(stats.not_in_window_pct) == 20.0

    assert pytest.approx(stats.dist_1_2_pct) == 40.0
    assert pytest.approx(stats.dist_3_plus_pct) == 20.0
    assert pytest.approx(stats.dist_0_pct) == 20.0
    assert pytest.approx(stats.dist_no_nearby_note_pct) == 20.0

