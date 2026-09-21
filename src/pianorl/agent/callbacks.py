"""Callback utilities for Piano-RL training."""

from typing import List, Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MultiKeyTensorboardCallback(BaseCallback):
    """Logs episode metrics and average keys pressed per step to TensorBoard."""

    def __init__(self, log_freq: int = 256):
        super().__init__()
        self.log_freq = log_freq
        self.step_keys_pressed: List[int] = []

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            # actions shape: (n_envs, 88)
            num_pressed = np.sum(np.asarray(actions) > 0, axis=-1)
            self.step_keys_pressed.extend(num_pressed.tolist())

        if len(self.step_keys_pressed) >= self.log_freq:
            avg_keys = float(np.mean(self.step_keys_pressed))
            self.logger.record("rollout/mean_keys_pressed_per_step", avg_keys)
            self.logger.record("custom/avg_keys_pressed_per_step", avg_keys)

            # Log mean episode reward if available from monitor
            ep_info_buffer = getattr(self.model, "ep_info_buffer", None)
            if ep_info_buffer and len(ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in ep_info_buffer]
                mean_r = float(np.mean(recent_rewards))
                self.logger.record("rollout/mean_episode_reward", mean_r)
                self.logger.record("custom/mean_episode_reward", mean_r)

            self.step_keys_pressed.clear()

        return True
