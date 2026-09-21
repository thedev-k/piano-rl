"""1D Pitch-Convolution Multi-Binary Actor-Critic Policy for Stable-Baselines3 PPO.

Adapts the shared 1D convolution reader across 88 piano keys to output independent
Bernoulli action logits for each key, enabling simultaneous multi-key strikes
(chords and polyphony) in Gymnasium MultiBinary(88) action spaces.
"""

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.distributions import BernoulliDistribution, Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from pianorl.score import NUM_PIANO_KEYS


class MultiKeyPitchConvPolicy(ActorCriticPolicy):
    """PPO Actor-Critic Policy for multi-key polyphony using shared 1D pitch convolutions.

    Action Head:
        Outputs 88 independent Bernoulli logits for Gymnasium MultiBinary(88) action spaces.
        Initial action biases are explicitly set to -3.0 so that newborn agents start
        with a sparse prior (~4.7% probability per key), preventing catastrophic
        shotgun/spam penalties on step 1.
    """

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
        """Construct the 1D convolution stack, multi-binary action head, and value head."""
        # 1. Per-key reader convolution stack
        # Input channels = 2 channels * 16 time slots (32) + 4 beat one-hot + 1 tempo = 37 channels
        self.conv1 = nn.Conv1d(in_channels=37, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # 2. Final action linear layer producing 1 logit per key (shared across 88 keys)
        self.action_net = nn.Linear(64, 1)

        # 3. Value function network (critic)
        # Global summary size = 64 (avg pool) + 64 (max pool) + 4 (beat pos) + 1 (tempo) = 133
        self.value_net = nn.Sequential(
            nn.Linear(133, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Action distribution: MultiBinary(88) with independent Bernoulli heads
        self.action_dist = BernoulliDistribution(NUM_PIANO_KEYS)

        # Weight and bias initialization
        if self.ortho_init:
            for m in [self.conv1, self.conv2, self.conv3]:
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            # Initialize action linear layer with small gain
            nn.init.orthogonal_(self.action_net.weight, gain=0.01)

            for m in self.value_net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.constant_(m.bias, 0.0)

        # Initialization Trick: Explicitly set action bias to -3.0
        # sigmoid(-3.0) ~= 0.0474 (4.7% probability of pressing each key initially)
        nn.init.constant_(self.action_net.bias, -3.0)

        # Optimizer setup
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

        total_params = sum(p.numel() for p in self.parameters())
        print(f"MultiKeyPitchConvPolicy initialized with {total_params:,} parameters.")

    def _forward_network(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """Compute 88 independent action logits and value estimate from observation tensor."""
        # Unpack observation
        # Window flat size = 2 * 88 * 16 = 2816
        window = (
            obs[:, :2816]
            .view(-1, 2, NUM_PIANO_KEYS, 16)
            .permute(0, 1, 3, 2)
            .reshape(-1, 32, NUM_PIANO_KEYS)
        )
        pos_in_beat = obs[:, 2816:2820]
        tempo = obs[:, 2820:2821]

        # Broadcast beat position and tempo as extra channels across all 88 keys
        extra = (
            th.cat([pos_in_beat, tempo], dim=-1)
            .unsqueeze(-1)
            .expand(-1, 5, NUM_PIANO_KEYS)
        )
        x = th.cat([window, extra], dim=1)  # (batch, 37, 88)

        # 1D Convolution stack along pitch axis
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))  # (batch, 64, 88)

        # Apply action linear layer across all 88 pitch positions:
        # (batch, 64, 88) -> (batch, 88, 64) -> action_net -> (batch, 88, 1) -> (batch, 88)
        action_logits = self.action_net(h.permute(0, 2, 1)).squeeze(-1)

        # Global summary for value function
        avg_pool = h.mean(dim=-1)
        max_pool = h.max(dim=-1).values
        summary = th.cat([avg_pool, max_pool, pos_in_beat, tempo], dim=-1)  # (batch, 133)
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
