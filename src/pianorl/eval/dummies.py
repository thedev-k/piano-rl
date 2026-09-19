import random
from typing import Optional


class PerfectPlayer:
    """A baseline player that peeks at the environment's true note targets

    and strikes every note at its exact scheduled step.
    """

    def act(self, env) -> int:
        for target in env.targets:
            if target["start_step"] == env.current_step:
                # pitch 21 maps to action 1, pitch 108 maps to action 88
                return target["pitch"] - 20
        return 0


class SilentPlayer:
    """A baseline player that never presses any keys (always does nothing)."""

    def act(self, env) -> int:
        return 0


class SpamPlayer:
    """A baseline player that wildly strikes a random piano key (1 to 88) at every step."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, env) -> int:
        return self.rng.randint(1, 88)
