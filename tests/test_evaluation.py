import json
from pathlib import Path
import numpy as np
import pytest

from pianorl.score import Score, NoteEvent, load_score
from pianorl.env import PianoFreeKeysEnv, split_observation
from pianorl.eval import (
    DoNothingPlayer,
    EpisodeCounters,
    EvaluationMetrics,
    RandomPlayer,
    RuleBasedPlayer,
    aggregate_and_compute,
    compute_metrics,
)
from scripts.evaluate import evaluate_player


def test_metrics_on_handmade_counters():
    """Check metric computations: perfect gives 1.0, silent gives 0.0, zero division returns 0.0."""
    # 1. Perfect play
    perfect_c = EpisodeCounters(
        hits_exact=10,
        hits_off_by_one=0,
        wrong_presses=0,
        missed_notes=0,
        total_notes=10,
        total_reward=10.0,
    )
    m_perf = compute_metrics(perfect_c, num_pieces=1)
    assert m_perf.precision == 1.0
    assert m_perf.recall == 1.0
    assert m_perf.f1 == 1.0
    assert m_perf.exact_rate == 1.0
    assert m_perf.mean_reward == 10.0

    # 2. Silent play (did nothing, missed all notes)
    silent_c = EpisodeCounters(
        hits_exact=0,
        hits_off_by_one=0,
        wrong_presses=0,
        missed_notes=10,
        total_notes=10,
        total_reward=-10.0,
    )
    m_silent = compute_metrics(silent_c, num_pieces=1)
    assert m_silent.precision == 0.0
    assert m_silent.recall == 0.0
    assert m_silent.f1 == 0.0
    assert m_silent.exact_rate == 0.0
    assert m_silent.mean_reward == -10.0

    # 3. Zero denominators (all zeros) must not crash
    empty_c = EpisodeCounters()
    m_empty = compute_metrics(empty_c, num_pieces=0)
    assert m_empty.precision == 0.0
    assert m_empty.recall == 0.0
    assert m_empty.f1 == 0.0
    assert m_empty.exact_rate == 0.0
    assert m_empty.mean_reward == 0.0


def test_grouping_aggregates_before_computing():
    """Verify grouping adds counters across pieces before computing metrics, not averaging percentages."""
    # Piece 1: 1 hit, 0 wrong presses -> precision = 1.0
    c1 = EpisodeCounters(hits_exact=1, wrong_presses=0, total_notes=1)
    # Piece 2: 10 hits, 10 wrong presses -> precision = 10 / 20 = 0.5
    c2 = EpisodeCounters(hits_exact=10, wrong_presses=10, total_notes=10)

    # Average of percentages would be (1.0 + 0.5) / 2 = 0.75
    # True pooled precision is (1 + 10) / (1 + 10 + 10) = 11 / 21 =~ 0.5238
    m = aggregate_and_compute([c1, c2])
    expected_precision = 11.0 / 21.0
    assert m.precision == pytest.approx(expected_precision, abs=1e-4)


def test_split_observation_shapes_and_values():
    """Verify split_observation cleanly reconstructs window, beat position, and tempo."""
    # Build dummy observation with known values
    n_beats = 4
    steps_per_beat = 4
    slots = 16
    window_shape = (2, 88, slots)
    window_size = 2 * 88 * slots

    window_sample = np.zeros(window_shape, dtype=np.float32)
    window_sample[0, 39, 0] = 1.0  # C4 onset at slot 0
    window_sample[1, 39, :4] = 1.0  # C4 held for 4 slots

    pos_sample = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    tempo_sample = np.array([0.5], dtype=np.float32)  # 100 BPM / 200

    flat_obs = np.concatenate([window_sample.flatten(), pos_sample, tempo_sample])

    w, pos, tempo = split_observation(flat_obs, n_beats=n_beats, steps_per_beat=steps_per_beat)

    assert w.shape == window_shape
    assert w[0, 39, 0] == 1.0
    assert np.array_equal(w[1, 39, :4], np.ones(4, dtype=np.float32))
    assert np.array_equal(pos, pos_sample)
    assert tempo == pytest.approx(0.5)


def test_rule_based_player_on_plain_observation():
    """RuleBasedPlayer must work when given only a plain numpy observation array."""
    player = RuleBasedPlayer(n_beats=4, steps_per_beat=4)
    slots = 16
    window_size = 2 * 88 * slots

    obs = np.zeros(window_size + 4 + 1, dtype=np.float32)

    # If no note is starting at slot 0, must return action 0
    assert player.act(obs) == 0

    # Put a note onset at C4 (pitch 60, row 39) at slot 0
    # C4 is row 39 in channel 0: index = (0 * 88 + 39) * slots + 0 = 39 * 16 = 624
    obs[39 * slots] = 1.0

    # Must return key 40 (row 39 + 1)
    action = player.act(obs)
    assert action == 40


def test_do_nothing_player_recall_is_zero():
    """DoNothingPlayer gets 0 recall on any non-empty piece."""
    score = Score(notes=[NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0)])
    env = PianoFreeKeysEnv(scores=score)
    obs, info = env.reset()
    player = DoNothingPlayer()

    terminated = False
    while not terminated:
        action = player.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    counters = EpisodeCounters(
        hits_exact=info["hits_exact"],
        hits_off_by_one=info["hits_off_by_one"],
        wrong_presses=info["wrong_presses"],
        missed_notes=info["missed_notes"],
        total_notes=info["total_notes"],
    )
    m = compute_metrics(counters)
    assert m.recall == 0.0


def test_random_player_f1_and_reproducibility():
    """RandomPlayer has low F1 (< 0.1) and identical seed produces identical actions."""
    # Seed consistency
    p1 = RandomPlayer(seed=123)
    p2 = RandomPlayer(seed=123)
    dummy_obs = np.zeros(2821, dtype=np.float32)
    actions1 = [p1.act(dummy_obs) for _ in range(20)]
    actions2 = [p2.act(dummy_obs) for _ in range(20)]
    assert actions1 == actions2

    # Low F1 evaluation on a sample piece
    score = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=float(i), duration_beats=1.0)
            for i in range(16)
        ]
    )
    env = PianoFreeKeysEnv(scores=score, seed=42)
    obs, info = env.reset()

    terminated = False
    while not terminated:
        action = p1.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    counters = EpisodeCounters(
        hits_exact=info["hits_exact"],
        hits_off_by_one=info["hits_off_by_one"],
        wrong_presses=info["wrong_presses"],
        missed_notes=info["missed_notes"],
        total_notes=info["total_notes"],
    )
    m = compute_metrics(counters)
    assert m.f1 < 0.1


def test_env_reset_piece_index_option():
    """Verify reset(options={'piece_index': i}) loads that exact piece."""
    scores = [
        Score(notes=[NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0)]),
        Score(notes=[NoteEvent(pitch=72, start_beat=0.0, duration_beats=1.0)]),
    ]
    env = PianoFreeKeysEnv(scores=scores)

    # Select piece 0
    obs0, info0 = env.reset(options={"piece_index": 0})
    assert env.current_score.notes[0].pitch == 60

    # Select piece 1
    obs1, info1 = env.reset(options={"piece_index": 1})
    assert env.current_score.notes[0].pitch == 72


def test_rule_based_player_heldout_f1_threshold():
    """RuleBasedPlayer reaches F1 >= 0.99 on every level of the held-out set."""
    manifest_file = Path("data/manifest.json")
    assert manifest_file.exists(), "data/manifest.json must exist"

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    heldout_items = [item for item in manifest if item["split"] == "heldout"]
    scores = [load_score(Path("data") / item["filename"]) for item in heldout_items]

    player = RuleBasedPlayer()
    results = evaluate_player(player, scores, heldout_items)

    for level in [1, 2, 3, 4]:
        lvl_name = f"Level {level}"
        assert lvl_name in results
        m = results[lvl_name]
        assert m.f1 >= 0.99, f"RuleBasedPlayer F1 below 0.99 on {lvl_name}: {m.f1}"
        assert m.exact_rate >= 0.99


def test_evaluate_visits_each_heldout_piece_exactly_once():
    """Verify that evaluate_player visits every held-out piece in the manifest."""
    manifest_file = Path("data/manifest.json")
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    heldout_items = [item for item in manifest if item["split"] == "heldout"]
    scores = [load_score(Path("data") / item["filename"]) for item in heldout_items]

    player = DoNothingPlayer()
    results = evaluate_player(player, scores, heldout_items)

    assert results["Overall"].num_pieces == len(heldout_items)
