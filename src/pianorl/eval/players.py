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


class SilentMultiPlayer:
    """Fair baseline player that never strikes any key in MultiKeyPianoEnv."""

    def __init__(self):
        self.is_multikey = True

    def act(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(88, dtype=np.int8)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.act(observation), None


class RandomMultiPlayer:
    """Fair baseline player that randomly strikes keys in MultiKeyPianoEnv.

    By default, each key has a ~5% probability of being struck, matching the
    prior distribution of our newborn policy network.
    """

    def __init__(self, prob: float = 0.05, seed: Optional[int] = None):
        self.is_multikey = True
        self.prob = prob
        self.rng = np.random.default_rng(seed)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return (self.rng.random(88) < self.prob).astype(np.int8)

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.act(observation), None



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


class PPOPlayer:
    """Fair player driven by a trained Stable-Baselines3 PPO model.

    Receives ONLY the observation vector and uses deterministic prediction.
    Supports both single-key Discrete(89) and multi-key MultiBinary(88) policies.
    """

    def __init__(self, model_path):
        from stable_baselines3 import PPO
        import gymnasium as gym
        from pianorl.agent import PitchConvPolicy, MultiKeyPitchConvPolicy

        custom_objects = {
            "PitchConvPolicy": PitchConvPolicy,
            "MultiKeyPitchConvPolicy": MultiKeyPitchConvPolicy,
        }
        self.model = PPO.load(str(model_path), device="cpu", custom_objects=custom_objects)
        self.is_multikey = isinstance(self.model.action_space, gym.spaces.MultiBinary) or (
            hasattr(self.model.action_space, "shape") and self.model.action_space.shape == (88,)
        )

    def act(self, observation: np.ndarray):
        action, _states = self.model.predict(observation, deterministic=True)
        if self.is_multikey:
            return np.asarray(action, dtype=np.int8)
        return int(action)

