import random
from typing import Optional
import numpy as np

from pianorl.env import split_observation


class RandomPlayer:
    """Fair baseline player that chooses a uniformly random action (0 to 88) at each step."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, observation: np.ndarray) -> int:
        return self.rng.randint(0, 88)


class DoNothingPlayer:
    """Fair baseline player that never strikes any key (always returns action 0)."""

    def act(self, observation: np.ndarray) -> int:
        return 0


class RuleBasedPlayer:
    """Fair baseline player that sight-reads using ONLY the incoming observation vector.

    It unflattens the ScoreWindow, inspects slot 0 (current step) of Channel 0
    (note onsets), and strikes any key scheduled to start right now.
    """

    def __init__(self, n_beats: int = 4, steps_per_beat: int = 4):
        self.n_beats = n_beats
        self.steps_per_beat = steps_per_beat

    def act(self, observation: np.ndarray) -> int:
        window, beat_pos, tempo = split_observation(
            observation, n_beats=self.n_beats, steps_per_beat=self.steps_per_beat
        )
        # Channel 0: note starts here; slot 0: current step
        active_onsets = np.where(window[0, :, 0] > 0.5)[0]
        if len(active_onsets) > 0:
            # Key action is 1-indexed (row 0 -> action 1)
            return int(active_onsets[0] + 1)
        return 0
