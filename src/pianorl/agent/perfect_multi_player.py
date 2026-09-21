"""Rule-based perfect baseline player for MultiKeyPianoEnv.

This agent reads the observation window (Channel 0: note onsets) and extracts
the notes scheduled to start at the current time step (slot 0). It outputs an
exact MultiBinary(88) action with 1 for each due note and 0 for all others,
achieving a perfect +1.0 reward per note with zero wrong strikes and zero misses.
"""

from typing import Optional, Tuple, Union
import numpy as np

from pianorl.env import split_observation
from pianorl.score import NUM_PIANO_KEYS


class PerfectMultiPlayer:
    """Fair rule-based baseline player for multi-key polyphonic piano practice.

    Inspects ONLY the incoming observation vector. Unflattens the ScoreWindow,
    reads slot 0 (current step) of Channel 0 (note onsets), and strikes all keys
    scheduled to start at this exact moment.
    """

    def __init__(self, n_beats: int = 4, steps_per_beat: int = 4):
        self.n_beats = n_beats
        self.steps_per_beat = steps_per_beat

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Decide which keys to press based solely on the observation vector.

        Args:
            observation: Flat observation array from MultiKeyPianoEnv.

        Returns:
            np.ndarray: Binary array of shape (88,) with dtype int8.
                1 indicates key strike, 0 indicates silence/no strike.
        """
        obs = np.asarray(observation)
        if obs.ndim > 1:
            obs = obs[0]

        window, beat_pos, tempo = split_observation(
            obs, n_beats=self.n_beats, steps_per_beat=self.steps_per_beat
        )
        # Channel 0: note onsets (1.0 where a note begins)
        # Slot 0: current time step
        current_step_onsets = window[0, :, 0]

        # MultiBinary(88): 1 for every key whose onset > 0.5, 0 otherwise
        action = (current_step_onsets > 0.5).astype(np.int8)
        return action

    def predict(
        self, observation: np.ndarray, state=None, episode_start=None, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compatible with Stable-Baselines3 model.predict interface."""
        obs = np.asarray(observation)
        if obs.ndim == 1:
            return self.act(obs), None
        # Batched observations
        actions = np.stack([self.act(o) for o in obs])
        return actions, None
