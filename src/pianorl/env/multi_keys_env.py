import random
from pathlib import Path
from typing import List, Optional, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pianorl.score import (
    Score,
    ScoreWindow,
    load_score,
    MIN_PIANO_PITCH,
    MAX_PIANO_PITCH,
    NUM_PIANO_KEYS,
)
from .free_keys_env import RewardConfig


class MultiKeyPianoEnv(gym.Env):
    """Gymnasium environment for polyphonic multi-key piano practice.

    In this environment, the agent can strike any combination of the 88 piano
    keys simultaneously at each 16th-note step (chords and polyphony), without
    physical hand or finger constraints yet.

    Action Space:
        MultiBinary(88):
            Array of 88 binary flags where index i (0..87) corresponds to
            piano key i + 1 (MIDI pitch 21 + i).
            1 = strike key at this time step
            0 = do not strike key

    Observation Space:
        Flat float32 Box vector matching PianoFreeKeysEnv:
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
        score_weights: Optional[List[float]] = None,
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

        if score_weights is not None:
            if len(score_weights) != len(self.scores):
                raise ValueError(
                    f"Length of score_weights ({len(score_weights)}) must match "
                    f"number of scores ({len(self.scores)})"
                )
            if any(w <= 0 for w in score_weights):
                raise ValueError("All score_weights must be strictly positive")
            self.score_weights = [float(w) for w in score_weights]
        else:
            self.score_weights = None

        self.n_beats = n_beats
        self.steps_per_beat = steps_per_beat
        self.num_slots = n_beats * steps_per_beat
        self.reward_config = reward_config if reward_config is not None else RewardConfig()

        self._rng = random.Random(seed)

        # Action space: 88 independent on/off key strikes
        self.action_space = spaces.MultiBinary(NUM_PIANO_KEYS)

        # Observation space matching PianoFreeKeysEnv
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
        window = self.window_engine.get_window(self.current_step)
        window_flat = window.flatten()

        pos_in_beat = self.current_step % self.steps_per_beat
        pos_one_hot = np.zeros(self.steps_per_beat, dtype=np.float32)
        pos_one_hot[pos_in_beat] = 1.0

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

        if options and "piece_index" in options:
            piece_idx = options["piece_index"]
            self.current_score = self.scores[piece_idx]
        else:
            if self.score_weights is not None:
                self.current_score = self._rng.choices(
                    self.scores, weights=self.score_weights, k=1
                )[0]
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
        seen_targets = set()
        for note in self.current_score.notes:
            start_step = int(round(note.start_beat * self.steps_per_beat))
            key = (note.pitch, start_step)
            if key in seen_targets:
                continue
            seen_targets.add(key)
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

        max_start = max((t["start_step"] for t in self.targets), default=0)
        self.end_step = max_start + 2 if self.targets else 0

        self.hits_exact = 0
        self.hits_off_by_one = 0
        self.wrong_presses = 0
        self.missed_notes = 0
        self.total_notes = len(self.targets)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: Union[np.ndarray, List[int]]):
        act_arr = np.asarray(action)
        if act_arr.shape != (NUM_PIANO_KEYS,):
            raise ValueError(
                f"Invalid action shape {act_arr.shape}; expected ({NUM_PIANO_KEYS},)"
            )

        t = self.current_step
        step_reward = 0.0

        # 1. Collect all struck pitches at this step
        # Index i corresponds to key i + 1 -> MIDI pitch 21 + i
        struck_pitches = [
            MIN_PIANO_PITCH + i for i in range(NUM_PIANO_KEYS) if act_arr[i] > 0
        ]

        unmatched_strikes = []

        # 2. Match exact timing hits first (target start_step == t)
        for pitch in struck_pitches:
            exact_match = None
            for target in self.targets:
                if (
                    not target["matched"]
                    and target["pitch"] == pitch
                    and target["start_step"] == t
                ):
                    exact_match = target
                    break

            if exact_match is not None:
                exact_match["matched"] = True
                step_reward += self.reward_config.hit_exact
                self.hits_exact += 1
            else:
                unmatched_strikes.append(pitch)

        # 3. Match off-by-one timing hits (|target start_step - t| == 1)
        still_unmatched_strikes = []
        for pitch in unmatched_strikes:
            off_match = None
            # Prioritize matching earlier note (t - 1) before late note (t + 1)
            for target in self.targets:
                if (
                    not target["matched"]
                    and target["pitch"] == pitch
                    and abs(target["start_step"] - t) == 1
                ):
                    off_match = target
                    break

            if off_match is not None:
                off_match["matched"] = True
                step_reward += self.reward_config.hit_off_by_one
                self.hits_off_by_one += 1
            else:
                still_unmatched_strikes.append(pitch)

        # 4. Any remaining unmatched strikes are extra/wrong presses
        for _ in still_unmatched_strikes:
            step_reward += self.reward_config.wrong_press
            self.wrong_presses += 1

        # 5. Check for newly missed notes
        # A note is missed if unmatched and current step is beyond its grace window (t >= start_step + 1)
        for target in self.targets:
            if not target["matched"] and not target["missed_penalized"]:
                if t >= target["start_step"] + 1:
                    target["missed_penalized"] = True
                    step_reward += self.reward_config.miss
                    self.missed_notes += 1

        # 6. Check termination (episode ends 2 steps after last note's start step)
        terminated = t >= self.end_step
        truncated = False

        # 7. Advance time
        self.current_step += 1

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(step_reward), bool(terminated), bool(truncated), info
