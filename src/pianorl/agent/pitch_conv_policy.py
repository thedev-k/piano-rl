"""1D Pitch-Convolution Actor-Critic Policy for Stable-Baselines3 PPO.

Instead of an MLP that treats each of the 88 piano keys independently with separate
weights, this policy applies a shared 1D convolution stack across the pitch dimension.
The exact same visual reading rule is evaluated across all 88 keys.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from pianorl.score import NUM_PIANO_KEYS


class PitchConvPolicy(ActorCriticPolicy):
    """PPO Actor-Critic Policy using shared 1D convolutions over the 88 piano keys."""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Any = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Construct the 1D convolution stack, action head, and value head."""
        # 1. Per-key reader convolution stack
        # Input channels = 2 channels * 16 time slots (32) + 4 beat one-hot + 1 tempo = 37 channels
        self.conv1 = nn.Conv1d(in_channels=37, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # 1x1 conv producing one logit per piano key (88 keys)
        self.key_conv = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)

        # Global summary size = 64 (avg pool) + 64 (max pool) + 4 (beat pos) + 1 (tempo) = 133
        # 2. No-op ("press nothing", action 0) network
        self.noop_net = nn.Sequential(
            nn.Linear(133, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # 3. Value function network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(133, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Action distribution: Discrete(89)
        self.action_dist = CategoricalDistribution(89)

        # Orthogonal initialization if enabled
        if self.ortho_init:
            for m in [self.conv1, self.conv2, self.conv3, self.key_conv]:
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            for m in self.noop_net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.constant_(m.bias, 0.0)
            for m in self.value_net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)

        # Optimizer setup
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"PitchConvPolicy initialized with {total_params:,} parameters.")

    def _forward_network(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Compute action logits (89 dims) and value estimate (1 dim) from observation tensor."""
        # Unpack observation
        # Window flat size = 2 * 88 * 16 = 2816
        # Reshape: (batch, 2, 88, 16) -> permute to (batch, 2, 16, 88) -> reshape to (batch, 32, 88)
        window = obs[:, :2816].view(-1, 2, NUM_PIANO_KEYS, 16).permute(0, 1, 3, 2).reshape(-1, 32, NUM_PIANO_KEYS)
        pos_in_beat = obs[:, 2816:2820]
        tempo = obs[:, 2820:2821]

        # Broadcast beat position and tempo as extra channels across all 88 keys
        extra = th.cat([pos_in_beat, tempo], dim=-1).unsqueeze(-1).expand(-1, 5, NUM_PIANO_KEYS)
        x = th.cat([window, extra], dim=1)  # (batch, 37, 88)

        # 1D Convolution stack along pitch axis
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))

        # Per-key logits: (batch, 1, 88) -> (batch, 88)
        key_logits = self.key_conv(h).squeeze(1)

        # Global summary for no-op logit and value function
        avg_pool = h.mean(dim=-1)
        max_pool = h.max(dim=-1).values
        summary = th.cat([avg_pool, max_pool, pos_in_beat, tempo], dim=-1)  # (batch, 133)

        noop_logit = self.noop_net(summary)  # (batch, 1)

        # Action 0 = no-op, Actions 1..88 = key strikes 1..88
        action_logits = th.cat([noop_logit, key_logits], dim=-1)  # (batch, 89)
        values = self.value_net(summary)  # (batch, 1)

        return action_logits, values

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_logits, values = self._forward_network(obs)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        action_logits, values = self._forward_network(obs)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        action_logits, _ = self._forward_network(obs)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        _, values = self._forward_network(obs)
        return values
