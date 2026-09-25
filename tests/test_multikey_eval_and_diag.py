"""Unit tests for multi-key evaluation, diagnostic error categorization, and baselines."""

import json
from pathlib import Path
import numpy as np
import pytest

from pianorl.agent import PerfectMultiPlayer
from pianorl.data import generate_multi_key_score
from pianorl.eval import (
    SilentMultiPlayer,
    RandomMultiPlayer,
    evaluate_multi_player,
    evaluate_multikey_real_piece,
)
from pianorl.eval.diagnosis_multikey import (
    categorize_multikey_wrong_press,
    run_multikey_diagnosis,
    run_multikey_midi_segment_diagnosis,
)
from pianorl.agent import MultiKeyTensorboardCallback


def test_evaluate_multi_baselines():
    """Verify evaluation metrics across Perfect, Silent, and Random baseline players."""
    scores = [
        generate_multi_key_score(1, seed=10),
        generate_multi_key_score(3, seed=30),  # contains 3-note chords
    ]
    items = [{"level": "1M"}, {"level": "3M"}]

    # 1. Perfect player: 1.0 F1, 100% exact rate, 100% chord exact rate
    perf_results = evaluate_multi_player(PerfectMultiPlayer(), scores, items)
    assert perf_results["Overall"].precision == 1.0
    assert perf_results["Overall"].recall == 1.0
    assert perf_results["Overall"].f1 == 1.0
    assert perf_results["Overall"].exact_rate == 1.0
    assert perf_results["Overall"].chord_exact_rate == 1.0

    # 2. Silent player: 0 hits, 0 precision, 0 recall
    silent_results = evaluate_multi_player(SilentMultiPlayer(), scores, items)
    assert silent_results["Overall"].precision == 0.0
    assert silent_results["Overall"].recall == 0.0
    assert silent_results["Overall"].f1 == 0.0
    assert silent_results["Overall"].exact_rate == 0.0
    assert silent_results["Overall"].mean_reward < 0.0

    # 3. Random player (p=5%): achieves low precision due to extra strikes
    random_player = RandomMultiPlayer(prob=0.05, seed=42)
    rand_results = evaluate_multi_player(random_player, scores, items)
    assert 0.0 <= rand_results["Overall"].precision <= 1.0
    assert 0.0 <= rand_results["Overall"].recall <= 1.0


def test_multikey_diagnosis_categorization():
    """Verify the 4 wrong-press categories: repeat, neighbor_key, not_in_window, no_note_nearby."""
    targets = [
        {"pitch": 60, "start_step": 10},  # Middle C at step 10
    ]
    # Observation window where only row 39 (pitch 60) has an onset at slot 0
    obs = np.zeros(2821, dtype=np.float32)
    # Channel 0, key 39 (pitch 60 = 21 + 39), slot 0
    obs[39 * 16] = 1.0

    # A. Repeat: pitch 60 struck at step 11 (within +-2 steps of target)
    cat_repeat = categorize_multikey_wrong_press(60, 11, targets, obs)
    assert cat_repeat == "repeat"

    # B. Neighbor key: pitch 61 struck at step 10 (1 semitone away from pitch 60)
    cat_neighbor = categorize_multikey_wrong_press(61, 10, targets, obs)
    assert cat_neighbor == "neighbor_key"

    # C. Not in window: pitch 72 (not active in window or nearby)
    cat_no_win = categorize_multikey_wrong_press(72, 10, targets, obs)
    assert cat_no_win == "not_in_window"

    # D. Put pitch 72 into window slot 8 (later in score), strike at step 10
    obs[51 * 16 + 8] = 1.0  # row 51 = pitch 72
    cat_no_note = categorize_multikey_wrong_press(72, 10, targets, obs)
    assert cat_no_note == "no_note_nearby"


def test_multikey_diagnosis_full_chords():
    """Verify run_multikey_diagnosis measures full-chord hit rate accurately."""
    score = generate_multi_key_score(3, seed=77)  # Triads
    items = [{"level": "3M"}]

    # Perfect player hits 100% of chords
    diag_perfect = run_multikey_diagnosis(PerfectMultiPlayer(), [score], items)
    assert diag_perfect["Overall"].chord_steps_total > 0
    assert diag_perfect["Overall"].chord_steps_all_hit == diag_perfect["Overall"].chord_steps_total
    assert diag_perfect["Overall"].full_chord_hit_rate == 1.0
    assert diag_perfect["Overall"].total_wrong == 0

    # Silent player hits 0% of chords
    diag_silent = run_multikey_diagnosis(SilentMultiPlayer(), [score], items)
    assert diag_silent["Overall"].chord_steps_all_hit == 0
    assert diag_silent["Overall"].full_chord_hit_rate == 0.0


def test_eval_real_multikey_piece():
    """Verify evaluate_multikey_real_piece steps a score without crashing."""
    real_file = Path("data/real/twinkle_twinkle.mid")
    if not real_file.exists():
        pytest.skip("data/real/twinkle_twinkle.mid not found")

    from pianorl.score import load_score
    score = load_score(real_file)

    player = PerfectMultiPlayer()
    counter, chord_cnt = evaluate_multikey_real_piece(player, score)
    assert counter.hits_exact == counter.total_notes
    assert counter.wrong_presses == 0
    assert counter.missed_notes == 0


def test_tensorboard_callback_logs_keys():
    """Verify MultiKeyTensorboardCallback correctly averages keys pressed per step."""
    cb = MultiKeyTensorboardCallback(log_freq=4)

    # Simulate 4 steps of actions
    actions_step1 = np.array([[1, 1, 0] + [0] * 85])  # 2 keys
    actions_step2 = np.array([[1, 0, 0] + [0] * 85])  # 1 key
    actions_step3 = np.array([[1, 1, 1] + [0] * 85])  # 3 keys
    actions_step4 = np.array([[0, 0, 0] + [0] * 85])  # 0 keys

    class MockLogger:
        def __init__(self):
            self.logged = {}
        def record(self, key, val):
            self.logged[key] = val

    class MockModel:
        def __init__(self, logger):
            self.logger = logger
            self.ep_info_buffer = []

    cb.model = MockModel(MockLogger())

    for act in [actions_step1, actions_step2, actions_step3, actions_step4]:
        cb.locals = {"actions": act}
        cb._on_step()

    # Average of [2, 1, 3, 0] is 1.5
    assert cb.model.logger.logged.get("rollout/mean_keys_pressed_per_step") == pytest.approx(1.5)
    assert cb.model.logger.logged.get("custom/avg_keys_pressed_per_step") == pytest.approx(1.5)


@pytest.mark.parametrize(
    "script_path",
    [
        "scripts/evaluate_multikey.py",
        "scripts/diagnose_multikey.py",
        "scripts/eval_real_multikey.py",
        "scripts/train_multikey.py",
    ],
)
def test_multikey_scripts_standalone_process(script_path: str):
    """Verify scripts start as separate processes from the project root without import errors."""
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, script_path, "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"{script_path} failed with returncode {res.returncode}:\n{res.stderr}"
    assert "ModuleNotFoundError" not in res.stderr
    assert "ImportError" not in res.stderr


def test_checkpoint_step_sorting():
    """Verify extract_checkpoint_step correctly parses step counts from filenames."""
    from scripts.evaluate_multikey import extract_checkpoint_step

    assert extract_checkpoint_step(Path("checkpoints/run/checkpoint_644800_steps.zip")) == 644800
    assert extract_checkpoint_step(Path("checkpoints/run/checkpoint_80000_steps.zip")) == 80000
    assert extract_checkpoint_step(Path("checkpoints/run/checkpoint_1000000_steps.zip")) == 1000000
    assert extract_checkpoint_step(Path("checkpoints/run/final.zip")) == 999999999

    paths = [
        Path("checkpoint_800000_steps.zip"),
        Path("checkpoint_644800_steps.zip"),
        Path("checkpoint_964800_steps.zip"),
        Path("final.zip"),
    ]
    paths.sort(key=extract_checkpoint_step)
    assert [p.name for p in paths] == [
        "checkpoint_644800_steps.zip",
        "checkpoint_800000_steps.zip",
        "checkpoint_964800_steps.zip",
        "final.zip",
    ]


def test_print_checkpoint_progression_table(capsys):
    """Verify progression summary table renders correctly without errors."""
    from scripts.evaluate_multikey import print_checkpoint_progression_table
    from pianorl.eval import MultiKeyEvaluationMetrics

    m1 = MultiKeyEvaluationMetrics(
        num_pieces=10,
        total_notes=100,
        precision=0.9,
        recall=0.85,
        f1=0.874,
        exact_rate=0.8,
        chord_exact_rate=0.95,
        mean_reward=15.0,
    )
    m2 = MultiKeyEvaluationMetrics(
        num_pieces=10,
        total_notes=100,
        precision=0.85,
        recall=0.80,
        f1=0.824,
        exact_rate=0.75,
        chord_exact_rate=0.88,
        mean_reward=12.0,
    )
    records = [
        {"name": "checkpoint_644800_steps", "overall": m1, "levels": {"1M": m1, "2M": m2}},
        {"name": "checkpoint_804800_steps", "overall": m2, "levels": {"1M": m2, "2M": m1}},
    ]

    print_checkpoint_progression_table(records, ["1M", "2M"], "heldout")
    captured = capsys.readouterr().out
    assert "MULTI-CHECKPOINT PROGRESSION SUMMARY" in captured
    assert "checkpoint_644800_steps" in captured
    assert "checkpoint_804800_steps" in captured
    assert "95.0%" in captured
    assert "88.0%" in captured


def test_run_multikey_midi_segment_diagnosis():
    """Verify run_multikey_midi_segment_diagnosis correctly splits piece into time segments."""
    score = generate_multi_key_score(3, seed=42)  # Triads
    player = PerfectMultiPlayer()

    # Split at 2.0 seconds
    results, returned_score = run_multikey_midi_segment_diagnosis(
        player_or_path=player,
        score_or_path=score,
        split_seconds=[2.0],
    )

    assert returned_score is score
    assert "Overall" in results
    seg_keys = [k for k in results.keys() if k != "Overall"]
    assert len(seg_keys) == 2
    assert seg_keys[0].startswith("0.0s - 2.0s")

    overall = results["Overall"]
    assert overall.total_notes > 0
    assert overall.hits_exact == overall.total_notes
    assert overall.total_wrong == 0
    assert overall.full_chord_hit_rate == 1.0

    # Ensure segment sums equal overall stats
    sum_notes = sum(results[k].total_notes for k in seg_keys)
    sum_hits = sum(results[k].hits_exact for k in seg_keys)
    sum_presses = sum(results[k].total_presses for k in seg_keys)
    assert sum_notes == overall.total_notes
    assert sum_hits == overall.hits_exact
    assert sum_presses == overall.total_presses


def test_print_multikey_midi_segment_table(capsys):
    """Verify print_multikey_midi_segment_table prints nicely formatted segment output."""
    from scripts.diagnose_multikey import print_multikey_midi_segment_table
    from pianorl.eval.diagnosis_multikey import MultiKeyDiagnosisStats

    s1 = MultiKeyDiagnosisStats(
        pieces_count=1,
        total_notes=10,
        total_presses=10,
        hits_exact=10,
        chord_steps_total=2,
        chord_steps_all_hit=2,
    )
    s2 = MultiKeyDiagnosisStats(
        pieces_count=1,
        total_notes=20,
        total_presses=22,
        hits_exact=18,
        wrong_neighbor_key=2,
        chord_steps_total=5,
        chord_steps_all_hit=4,
    )
    overall = s1.add(s2)

    results = {
        "0.0s - 30.0s": s1,
        "30.0s - 60.0s": s2,
        "Overall": overall,
    }

    print_multikey_midi_segment_table(
        results=results,
        model_path=Path("final.zip"),
        midi_path=Path("rush_e.mid"),
        duration_seconds=60.0,
        total_notes=30,
    )

    captured = capsys.readouterr().out
    assert "MULTI-KEY MIDI DIAGNOSTIC REPORT" in captured
    assert "0.0s - 30.0s" in captured
    assert "30.0s - 60.0s" in captured
    assert "Overall" in captured
    assert "rush_e.mid" in captured



