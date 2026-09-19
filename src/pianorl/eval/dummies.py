import random
from typing import Optional

# ==============================================================================
# NOTE: These dummy players peek into environment internals (e.g. env.targets).
# They are test tools only, never fair players!
# For fair players that observe only the observation vector, see players.py.
# ==============================================================================


class PerfectPlayer:
    """A baseline player that peeks at the environment's true note targets
    and strikes every note at its exact scheduled step.
    (Test tool only, never a fair player).
    """

    def act(self, env) -> int:
        for target in env.targets:
            if target["start_step"] == env.current_step:
                # pitch 21 maps to action 1, pitch 108 maps to action 88
                return target["pitch"] - 20
        return 0


class SilentPlayer:
    """A baseline player that never presses any keys (always does nothing).
    (Test tool only, never a fair player).
    """

    def act(self, env) -> int:
        return 0


class SpamPlayer:
    """A baseline player that wildly strikes a random piano key (1 to 88) at every step.
    (Test tool only, never a fair player).
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, env) -> int:
        return self.rng.randint(1, 88)
