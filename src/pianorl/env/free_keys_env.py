import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pianorl.score import Score, ScoreWindow, load_score, MIN_PIANO_PITCH, MAX_PIANO_PITCH, NUM_PIANO_KEYS


@dataclass
class RewardConfig:
    """Scoring point values for the Free-Keys practice environment."""
    hit_exact: float = 1.0          # Striking correct key on exact start step
    hit_off_by_one: float = 0.5     # Striking correct key 1 step early or late
    wrong_press: float = -0.5       # Striking a key with no matching note nearby
    miss: float = -1.0              # Failing to strike a note before window closes
    neighbor_key_penalty: float = 0.0  # Extra penalty when an unmatched key is within 1-2 semitones of an active note


def split_observation(
    obs: np.ndarray,
    n_beats: int = 4,
    steps_per_beat: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split a flat observation vector back into its constituent components.

    Args:
        obs: Flat 1D numpy array from PianoFreeKeysEnv.
        n_beats: Lookahead length in beats (default: 4).
        steps_per_beat: Resolution in steps per beat (default: 4).

    Returns:
        Tuple containing:
            - window: np.ndarray of shape (2, 88, slots)
            - beat_position: np.ndarray of shape (steps_per_beat,) (one-hot vector)
            - tempo: float (normalized tempo in [0, 1])
    """
    slots = n_beats * steps_per_beat
    window_size = 2 * NUM_PIANO_KEYS * slots

    window_flat = obs[:window_size]
    window = window_flat.reshape((2, NUM_PIANO_KEYS, slots))

    beat_position = obs[window_size : window_size + steps_per_beat]
    tempo = float(obs[window_size + steps_per_beat])

    return window, beat_position, tempo


class PianoFreeKeysEnv(gym.Env):

    """Gymnasium environment for free-keys piano practice.

    In this environment, physical hands and fingers are not modeled yet. The
    agent is free to strike any of the 88 piano keys (or do nothing) at each 16th-note step.

    Action Space:
        Discrete(89):
            0: Do nothing (rest)
            1 to 88: Strike piano key (1 = pitch 21 / A0, 88 = pitch 108 / C8)

    Observation Space:
        Flat float32 Box vector:
            - Flattened ScoreWindow: 2 channels x 88 keys x num_slots
            - Beat position: One-hot vector of length steps_per_beat
            - Normalized tempo: tempo_bpm / 200.0
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scores: Union[Score, str, Path, List[Union[Score, str, Path]]],
        n_beats: int = 4,
        steps_per_beat: int = 4,
        seed: Optional[int] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        super().__init__()

        # Ensure scores is a list of Score objects
        if not isinstance(scores, list):
            scores = [scores]

        self.scores: List[Score] = []
        for s in scores:
            if isinstance(s, (str, Path)):
                self.scores.append(load_score(s))
            elif isinstance(s, Score):
                self.scores.append(s)
            else:
                raise TypeError(f"Expected Score or path, got {type(s)}")

        if not self.scores:
            raise ValueError("Must provide at least one Score object or file path")

        self.n_beats = n_beats
        self.steps_per_beat = steps_per_beat
        self.num_slots = n_beats * steps_per_beat
        self.reward_config = reward_config if reward_config is not None else RewardConfig()

        self._rng = random.Random(seed)

        # Action space: 0 = rest, 1..88 = key strikes
        self.action_space = spaces.Discrete(89)

        # Observation dimension
        window_size = 2 * NUM_PIANO_KEYS * self.num_slots
        obs_dim = window_size + self.steps_per_beat + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Runtime state
        self.current_score: Optional[Score] = None
        self.window_engine: Optional[ScoreWindow] = None
        self.current_step: int = 0
        self.end_step: int = 0
        self.targets: List[dict] = []

        # Episode statistics
        self.hits_exact: int = 0
        self.hits_off_by_one: int = 0
        self.wrong_presses: int = 0
        self.missed_notes: int = 0
        self.total_notes: int = 0

    def _get_obs(self) -> np.ndarray:
        """Construct the single flat float32 observation vector."""
        # 1. Flattened score window
        window = self.window_engine.get_window(self.current_step)
        window_flat = window.flatten()

        # 2. One-hot position within beat
        pos_in_beat = self.current_step % self.steps_per_beat
        pos_one_hot = np.zeros(self.steps_per_beat, dtype=np.float32)
        pos_one_hot[pos_in_beat] = 1.0

        # 3. Normalized tempo (clamped to [0, 1])
        tempo_norm = np.array(
            [min(1.0, max(0.0, self.current_score.tempo_bpm / 200.0))],
            dtype=np.float32,
        )

        obs = np.concatenate([window_flat, pos_one_hot, tempo_norm]).astype(np.float32)
        return obs

    def _get_info(self) -> dict:
        """Return running counters and state information."""
        return {
            "hits_exact": self.hits_exact,
            "hits_off_by_one": self.hits_off_by_one,
            "wrong_presses": self.wrong_presses,
            "missed_notes": self.missed_notes,
            "total_notes": self.total_notes,
            "current_step": self.current_step,
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = random.Random(seed)

        # Pick piece: specific index if requested, otherwise random choice
        if options and "piece_index" in options:
            piece_idx = options["piece_index"]
            self.current_score = self.scores[piece_idx]
        else:
            self.current_score = self._rng.choice(self.scores)
        self.window_engine = ScoreWindow(
            self.current_score,
            n_beats=self.n_beats,
            steps_per_beat=self.steps_per_beat,
        )
        self.current_step = 0

        # Setup note targets for evaluation
        self.targets = []
        for note in self.current_score.notes:
            start_step = int(round(note.start_beat * self.steps_per_beat))
            self.targets.append(
                {
                    "pitch": note.pitch,
                    "start_step": start_step,
                    "duration_beats": note.duration_beats,
                    "velocity": getattr(note, "velocity", 80),
                    "matched": False,
                    "missed_penalized": False,
                }
            )

        # Episode ends 2 steps after last note's start step
        max_start = max((t["start_step"] for t in self.targets), default=0)
        self.end_step = max_start + 2 if self.targets else 0

        # Reset episode counters
        self.hits_exact = 0
        self.hits_off_by_one = 0
        self.wrong_presses = 0
        self.missed_notes = 0
        self.total_notes = len(self.targets)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; must be in Discrete(89)")

        t = self.current_step
        step_reward = 0.0

        # 1. Process agent key strike
        if action > 0:
            pitch = (MIN_PIANO_PITCH - 1) + action  # action 1 -> 21, action 88 -> 108

            # Look for an unmatched note within [t-1, t+1]
            match_target = None

            # Prioritize exact timing match first
            for target in self.targets:
                if not target["matched"] and target["pitch"] == pitch and target["start_step"] == t:
                    match_target = target
                    break

            # If no exact match, check for off-by-one match
            if match_target is None:
                for target in self.targets:
                    if not target["matched"] and target["pitch"] == pitch and abs(target["start_step"] - t) == 1:
                        match_target = target
                        break

            if match_target is not None:
                match_target["matched"] = True
                if match_target["start_step"] == t:
                    step_reward += self.reward_config.hit_exact
                    self.hits_exact += 1
                else:
                    step_reward += self.reward_config.hit_off_by_one
                    self.hits_off_by_one += 1
            else:
                step_reward += self.reward_config.wrong_press
                self.wrong_presses += 1

        # 2. Check for newly missed notes
        # A note is missed if still unmatched after step (start_step + 1) has been processed
        for target in self.targets:
            if not target["matched"] and not target["missed_penalized"]:
                if t >= target["start_step"] + 1:
                    target["missed_penalized"] = True
                    step_reward += self.reward_config.miss
                    self.missed_notes += 1

        # 3. Check termination (episode ends 2 steps after last note's start step)
        terminated = (t >= self.end_step)
        truncated = False

        # 4. Advance time
        self.current_step += 1

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(step_reward), bool(terminated), bool(truncated), info
