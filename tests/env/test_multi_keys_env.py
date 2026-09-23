import numpy as np
import pytest
from gymnasium import spaces

from pianorl.env import MultiKeyPianoEnv, PianoFreeKeysEnv, RewardConfig
from pianorl.score import NoteEvent, Score


def make_test_score(notes, tempo=120.0):
    return Score(notes=notes, tempo_bpm=tempo)


def make_action(pitches):
    """Helper to create an 88-element binary action array for given MIDI pitches."""
    action = np.zeros(88, dtype=np.int8)
    for p in pitches:
        assert 21 <= p <= 108, f"Pitch {p} outside 88-key range"
        action[p - 21] = 1
    return action


def test_action_and_observation_spaces():
    """Verify action space is MultiBinary(88) and observation space matches PianoFreeKeysEnv."""
    score = make_test_score([NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0)])
    env_multi = MultiKeyPianoEnv(scores=[score])
    env_single = PianoFreeKeysEnv(scores=[score])

    assert isinstance(env_multi.action_space, spaces.MultiBinary)
    assert env_multi.action_space.shape == (88,)
    assert env_multi.observation_space.shape == env_single.observation_space.shape

    # Invalid action shape should raise ValueError
    with pytest.raises(ValueError, match="Invalid action shape"):
        env_multi.step(np.zeros(50))


def test_exact_chord_match():
    """Verify striking all notes of a chord at the exact step awards +1.0 per note."""
    # 3-note C-major chord starting at beat 1.0 (step 4 in 16th-note steps)
    chord_notes = [
        NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0),
        NoteEvent(pitch=64, start_beat=1.0, duration_beats=1.0),
        NoteEvent(pitch=67, start_beat=1.0, duration_beats=1.0),
    ]
    score = make_test_score(chord_notes)
    env = MultiKeyPianoEnv(scores=[score])
    env.reset(options={"piece_index": 0})

    rest = make_action([])

    # Steps 0 to 3: rest (no notes playing)
    for _ in range(4):
        obs, reward, terminated, truncated, info = env.step(rest)
        assert reward == 0.0

    # Step 4: Strike the 3-note chord simultaneously (pitches 60, 64, 67)
    action_chord = make_action([60, 64, 67])
    obs, reward, terminated, truncated, info = env.step(action_chord)

    # 3 notes matched exactly -> +3.0 reward
    assert reward == pytest.approx(3.0)
    assert info["hits_exact"] == 3
    assert info["hits_off_by_one"] == 0
    assert info["wrong_presses"] == 0
    assert info["missed_notes"] == 0


def test_partial_chord_match():
    """Verify matching only part of a chord awards hits for played notes and misses for unplayed notes."""
    # 3-note chord at step 4: pitches 60, 64, 67
    chord_notes = [
        NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0),
        NoteEvent(pitch=64, start_beat=1.0, duration_beats=1.0),
        NoteEvent(pitch=67, start_beat=1.0, duration_beats=1.0),
    ]
    score = make_test_score(chord_notes)
    env = MultiKeyPianoEnv(scores=[score])
    env.reset(options={"piece_index": 0})

    rest = make_action([])
    for _ in range(4):
        env.step(rest)

    # Step 4: Agent strikes only 60 and 64 (leaves 67 unplayed)
    action_partial = make_action([60, 64])
    obs, reward, terminated, truncated, info = env.step(action_partial)

    # 2 exact hits awarded
    assert reward == pytest.approx(2.0)
    assert info["hits_exact"] == 2
    assert info["wrong_presses"] == 0
    assert info["missed_notes"] == 0

    # Step 5: Agent rests. Step 5 is >= start_step + 1 for key 67, so key 67 is now missed
    obs, reward, terminated, truncated, info = env.step(rest)
    assert reward == pytest.approx(-1.0)  # miss penalty
    assert info["missed_notes"] == 1


def test_extra_speculative_presses_hedging():
    """Verify extra/speculative strikes that don't match any note are penalized (-0.5 each)."""
    # 2-note dyad at step 4: pitches 60, 64
    dyad_notes = [
        NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0),
        NoteEvent(pitch=64, start_beat=1.0, duration_beats=1.0),
    ]
    score = make_test_score(dyad_notes)
    env = MultiKeyPianoEnv(scores=[score])
    env.reset(options={"piece_index": 0})

    rest = make_action([])
    for _ in range(4):
        env.step(rest)

    # Step 4: Agent strikes 2 correct keys (60, 64) PLUS 3 speculative wrong keys (62, 70, 72)
    action_hedged = make_action([60, 64, 62, 70, 72])
    obs, reward, terminated, truncated, info = env.step(action_hedged)

    # Reward: 2 * (+1.0) + 3 * (-0.5) = +2.0 - 1.5 = +0.5
    assert reward == pytest.approx(0.5)
    assert info["hits_exact"] == 2
    assert info["wrong_presses"] == 3
    assert info["missed_notes"] == 0


def test_all_keys_shotgun_penalty():
    """Verify that pressing all 88 keys simultaneously results in a catastrophic penalty."""
    score = make_test_score([NoteEvent(pitch=60, start_beat=1.0, duration_beats=1.0)])
    env = MultiKeyPianoEnv(scores=[score])
    env.reset(options={"piece_index": 0})

    rest = make_action([])
    for _ in range(4):
        env.step(rest)

    # Step 4: Press ALL 88 keys
    action_all = np.ones(88, dtype=np.int8)
    obs, reward, terminated, truncated, info = env.step(action_all)

    # 1 exact hit (+1.0) + 87 wrong presses (-0.5 * 87 = -43.5) = -42.5
    assert reward == pytest.approx(-42.5)
    assert info["hits_exact"] == 1
    assert info["wrong_presses"] == 87


def test_off_by_one_multi_key_match():
    """Verify striking multiple notes 1 step early or late awards +0.5 per note."""
    # 2-note chord at step 5: pitches 60, 67
    notes = [
        NoteEvent(pitch=60, start_beat=1.25, duration_beats=1.0),
        NoteEvent(pitch=67, start_beat=1.25, duration_beats=1.0),
    ]
    score = make_test_score(notes)
    env = MultiKeyPianoEnv(scores=[score])
    env.reset(options={"piece_index": 0})

    rest = make_action([])
    for _ in range(4):
        env.step(rest)

    # Step 4: Strike both keys 1 step early (step 4 instead of step 5)
    action_early = make_action([60, 67])
    obs, reward, terminated, truncated, info = env.step(action_early)

    # Both are off-by-one matches: 2 * (+0.5) = +1.0
    assert reward == pytest.approx(1.0)
    assert info["hits_off_by_one"] == 2
    assert info["hits_exact"] == 0
    assert info["wrong_presses"] == 0

    # Step 5 & 6: Agent rests. Since notes were already matched, no miss penalty occurs
    obs, reward_5, _, _, info = env.step(rest)
    obs, reward_6, _, _, info = env.step(rest)
    assert reward_5 == 0.0
    assert reward_6 == 0.0
    assert info["missed_notes"] == 0


def test_episode_lifecycle_and_termination():
    """Verify episode runs to completion, terminates properly, and returns valid obs shapes."""
    notes = [
        NoteEvent(pitch=60, start_beat=0.0, duration_beats=1.0),
        NoteEvent(pitch=64, start_beat=0.0, duration_beats=1.0),
        NoteEvent(pitch=67, start_beat=1.0, duration_beats=1.0),
    ]
    score = make_test_score(notes)
    env = MultiKeyPianoEnv(scores=[score])
    obs, info = env.reset(options={"piece_index": 0})

    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert info["total_notes"] == 3

    step_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.zeros(88, dtype=np.int8))
        step_count += 1
        assert obs.shape == env.observation_space.shape

    assert terminated
    assert info["missed_notes"] == 3
    assert step_count == env.end_step + 1


def test_neighbor_key_penalty():
    """Verify that neighbor key strikes (1-2 semitones off) receive additional penalty when configured."""
    # Target note at beat 1.0 (step 4): pitch 60
    notes = [NoteEvent(pitch=60, start_beat=1.0, duration_beats=2.0)]
    score = make_test_score(notes)

    # 1. Default RewardConfig (neighbor_key_penalty = 0.0): both neighbor and far key get -0.5
    env_default = MultiKeyPianoEnv(scores=[score])
    env_default.reset(options={"piece_index": 0})
    for _ in range(4):
        env_default.step(make_action([]))

    # Strike correct 60 + neighbor 61 (1 semitone) + far 75 (15 semitones)
    act = make_action([60, 61, 75])
    _, reward, _, _, info = env_default.step(act)
    # +1.0 (exact 60) + -0.5 (wrong 61) + -0.5 (wrong 75) = 0.0
    assert reward == pytest.approx(0.0)
    assert info["wrong_presses"] == 2

    # 2. Configured RewardConfig with neighbor_key_penalty = -0.3
    cfg_penalty = RewardConfig(wrong_press=-0.5, neighbor_key_penalty=-0.3)
    env_penalty = MultiKeyPianoEnv(scores=[score], reward_config=cfg_penalty)
    env_penalty.reset(options={"piece_index": 0})
    for _ in range(4):
        env_penalty.step(make_action([]))

    # A) Far wrong key (pitch 75, distance 15): gets standard -0.5 penalty
    act_far = make_action([60, 75])
    _, reward_far, _, _, _ = env_penalty.step(act_far)
    # +1.0 (exact 60) - 0.5 (far wrong 75) = +0.5
    assert reward_far == pytest.approx(0.5)

    # Reset and test B) Neighbor wrong key (pitch 61, distance 1 semitone): gets -0.5 + -0.3 = -0.8 penalty
    env_penalty.reset(options={"piece_index": 0})
    for _ in range(4):
        env_penalty.step(make_action([]))

    act_neighbor = make_action([60, 61])
    _, reward_neighbor, _, _, _ = env_penalty.step(act_neighbor)
    # +1.0 (exact 60) - 0.8 (neighbor wrong 61) = +0.2
    assert reward_neighbor == pytest.approx(0.2)

    # Reset and test C) Distance 2 semitones (pitch 62): also neighbor key -> -0.8 penalty
    env_penalty.reset(options={"piece_index": 0})
    for _ in range(4):
        env_penalty.step(make_action([]))

    act_neighbor_2 = make_action([60, 62])
    _, reward_neighbor_2, _, _, _ = env_penalty.step(act_neighbor_2)
    # +1.0 (exact 60) - 0.8 (neighbor wrong 62) = +0.2
    assert reward_neighbor_2 == pytest.approx(0.2)

