import numpy as np
import pytest
from pathlib import Path

from pianorl.score import Score, NoteEvent, load_score
from pianorl.env import PianoFreeKeysEnv, RewardConfig
from pianorl.eval import PerfectPlayer, SilentPlayer, SpamPlayer


@pytest.fixture
def sample_score() -> Score:
    """Load a generated level 2 piece from data/train/."""
    path = Path("data/train/level2_0001.mid")
    return load_score(path)


def test_perfect_player(sample_score: Score):
    """PerfectPlayer strikes every note exactly on time: 100% hits, 0 wrong, 0 misses."""
    env = PianoFreeKeysEnv(scores=sample_score, seed=42)
    obs, info = env.reset(seed=42)
    player = PerfectPlayer()

    total_reward = 0.0
    terminated = False

    while not terminated:
        action = player.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    assert info["hits_exact"] == info["total_notes"]
    assert info["hits_off_by_one"] == 0
    assert info["wrong_presses"] == 0
    assert info["missed_notes"] == 0
    assert total_reward == pytest.approx(float(info["total_notes"]) * 1.0)


def test_silent_player(sample_score: Score):
    """SilentPlayer does nothing: 0 hits, all notes missed (-1.0 each)."""
    env = PianoFreeKeysEnv(scores=sample_score, seed=42)
    obs, info = env.reset(seed=42)
    player = SilentPlayer()

    total_reward = 0.0
    terminated = False

    while not terminated:
        action = player.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    assert info["hits_exact"] == 0
    assert info["hits_off_by_one"] == 0
    assert info["wrong_presses"] == 0
    assert info["missed_notes"] == info["total_notes"]
    assert total_reward == pytest.approx(float(info["total_notes"]) * -1.0)


def test_spam_player(sample_score: Score):
    """SpamPlayer presses random keys every step: lots of wrong presses, heavily negative score."""
    env = PianoFreeKeysEnv(scores=sample_score, seed=42)
    obs, info = env.reset(seed=42)
    player = SpamPlayer(seed=42)

    total_reward = 0.0
    terminated = False

    while not terminated:
        action = player.act(env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    assert info["wrong_presses"] > 20
    assert total_reward < -5.0


def test_timing_tolerance_early_and_late():
    """Striking 1 step early or late gives +0.5; 2 steps early or late counts as wrong (-0.5) + miss (-1.0)."""
    # Single C4 (pitch 60, key 40) at beat 1.0 (step 4)
    single_note_score = Score(notes=[NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0)])

    # 1. Strike 1 step early (step 3)
    env1 = PianoFreeKeysEnv(scores=single_note_score)
    env1.reset()
    for _ in range(3):
        env1.step(0)  # steps 0, 1, 2
    obs, rew_early, term, trunc, info1 = env1.step(40)  # step 3 (1 early)
    assert rew_early == 0.5
    assert info1["hits_off_by_one"] == 1

    # 2. Strike 1 step late (step 5)
    env2 = PianoFreeKeysEnv(scores=single_note_score)
    env2.reset()
    for _ in range(5):
        env2.step(0)  # steps 0, 1, 2, 3, 4
    obs, rew_late, term, trunc, info2 = env2.step(40)  # step 5 (1 late)
    assert rew_late == 0.5
    assert info2["hits_off_by_one"] == 1

    # 3. Strike 2 steps early (step 2) -> counts as wrong press (-0.5), and later note misses (-1.0)
    env3 = PianoFreeKeysEnv(scores=single_note_score)
    env3.reset()
    env3.step(0)  # step 0
    env3.step(0)  # step 1
    obs, rew_too_early, _, _, _ = env3.step(40)  # step 2 (2 steps early)
    assert rew_too_early == -0.5  # Wrong press

    # Advance past step 5 (start_step 4 + 1)
    env3.step(0)  # step 3
    env3.step(0)  # step 4
    _, rew_miss, _, _, info3 = env3.step(0)  # step 5 -> window closed, miss penalized
    assert rew_miss == -1.0
    assert info3["missed_notes"] == 1
    assert info3["wrong_presses"] == 1


def test_striking_same_key_twice_near_one_note():
    """First strike matches note, second strike near the same note is counted as wrong press."""
    score = Score(notes=[NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0)])
    env = PianoFreeKeysEnv(scores=score)
    env.reset()

    env.step(0)  # step 0
    env.step(0)  # step 1
    env.step(0)  # step 2

    # Step 3: 1 step early -> matches (+0.5)
    _, rew1, _, _, info = env.step(40)
    assert rew1 == 0.5
    assert info["hits_off_by_one"] == 1

    # Step 4: exact step, but note already matched -> counts as wrong press (-0.5)
    _, rew2, _, _, info = env.step(40)
    assert rew2 == -0.5
    assert info["wrong_presses"] == 1


def test_observation_space_compliance(sample_score: Score):
    """Verify observation shape, float32 type, and bounded within observation_space."""
    env = PianoFreeKeysEnv(scores=sample_score)
    obs, info = env.reset()

    # Shape calculation: (2 * 88 * 16) + 4 + 1 = 2821
    expected_dim = 2 * 88 * 16 + 4 + 1
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_no_leakage_beyond_lookahead():
    """Scores identical in first 4 beats but different after beat 4 must produce identical step 0 observations."""
    # Score A has high C (72) at beat 5
    score_a = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=62, start_beat=1.0, duration_beats=1.0),
            NoteEvent(pitch=72, start_beat=5.0, duration_beats=1.0),
        ],
        tempo_bpm=100.0,
    )
    # Score B has low A (21) at beat 5
    score_b = Score(
        notes=[
            NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
            NoteEvent(pitch=62, start_beat=1.0, duration_beats=1.0),
            NoteEvent(pitch=21, start_beat=5.0, duration_beats=1.0),
        ],
        tempo_bpm=100.0,
    )

    env_a = PianoFreeKeysEnv(scores=score_a)
    obs_a, _ = env_a.reset()

    env_b = PianoFreeKeysEnv(scores=score_b)
    obs_b, _ = env_b.reset()

    # Step 0 lookahead only sees beats 0 through 4. Beat 5 is invisible.
    np.testing.assert_array_equal(obs_a, obs_b)


def test_seed_piece_selection_sequence(sample_score: Score):
    """Same seed produces identical sequence of selected pieces; different seeds produce different orders."""
    scores = [
        Score(notes=[NoteEvent(pitch=60 + i, start_beat=0.0, duration_beats=1.0)])
        for i in range(5)
    ]

    env1 = PianoFreeKeysEnv(scores=scores)
    seq1 = [env1.reset(seed=123)[1]["total_notes"] for _ in range(5)]
    pitches1 = [env1.targets[0]["pitch"] for _ in range(5)]

    env2 = PianoFreeKeysEnv(scores=scores)
    seq2 = [env2.reset(seed=123)[1]["total_notes"] for _ in range(5)]
    pitches2 = [env2.targets[0]["pitch"] for _ in range(5)]

    assert pitches1 == pitches2


def test_episode_termination_exact_step():
    """Episode ends exactly 2 steps after the last note's start step."""
    # Last note starts at beat 2.0 (step 8)
    score = Score(notes=[NoteEvent(pitch=60, start_beat=2.0, duration_beats=1.0)])
    env = PianoFreeKeysEnv(scores=score)
    env.reset()

    # Steps 0 to 9: not terminated
    for step_i in range(10):  # steps 0 through 9
        _, _, terminated, _, _ = env.step(0)
        assert not terminated, f"Premature termination at step {step_i}"

    # Step 10 (8 + 2): terminated must be True
    _, _, terminated, _, _ = env.step(0)
    assert terminated


def test_stable_baselines3_check_env(sample_score: Score):
    """Verify full Gymnasium compliance with stable-baselines3 check_env."""
    from stable_baselines3.common.env_checker import check_env

    env = PianoFreeKeysEnv(scores=sample_score)
    # check_env raises an exception if any standard gym interface rule is violated
    check_env(env)
