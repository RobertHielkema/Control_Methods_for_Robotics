import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO controlling the UR7e.

    Observation:
        [q(6), q_dot(6), ee_position(3), target_position(3)]
        -> 18 dimensions

    Action:
        6 normalized joint torques in [-1, 1]

    The normalized action is converted to physical torque
    outside the network:
        torque = action * torque_limits
    """

    def __init__(self, obs_dim: int, action_dim: int):

        super().__init__()

        # ============================================================
        # Shared feature network
        # ============================================================

        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),

            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
        )

        # ============================================================
        # Actor
        # ============================================================

        self.actor_mean = layer_init(
            nn.Linear(256, action_dim),
            std=0.01
        )

        # Learnable exploration standard deviation
        self.actor_log_std = nn.Parameter(
            torch.ones(action_dim) * -0.5
        )

        # ============================================================
        # Critic
        # ============================================================

        self.critic = nn.Sequential(
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),

            layer_init(nn.Linear(256, 1), std=1.0)
        )

    # ================================================================
    # Feature extraction
    # ================================================================

    def _features(self, obs):

        return self.shared(obs)

    # ================================================================
    # Critic
    # ================================================================

    def get_value(self, obs):

        features = self._features(obs)

        value = self.critic(features)

        return value.squeeze(-1)

    # ================================================================
    # Actor + Critic
    # ================================================================

    def get_action_and_value(
        self,
        obs,
        action=None
    ):

        features = self._features(obs)

        # ------------------------------------------------------------
        # Actor
        # ------------------------------------------------------------

        action_mean = self.actor_mean(features)

        action_log_std = self.actor_log_std.expand_as(action_mean)

        action_std = torch.exp(action_log_std)

        dist = Normal(
            action_mean,
            action_std
        )

        # ------------------------------------------------------------
        # Sample action
        # ------------------------------------------------------------

        if action is None:

            # Sample from Gaussian
            raw_action = dist.rsample()

            # Squash to [-1, 1]
            action = torch.tanh(raw_action)

        else:

            # During PPO update we receive the stored action,
            # which is already in [-1, 1].
            #
            # Recover the corresponding pre-tanh value.
            action = torch.clamp(
                action,
                -0.999999,
                0.999999
            )

            raw_action = torch.atanh(action)

        # ------------------------------------------------------------
        # Log probability
        # ------------------------------------------------------------

        log_prob = dist.log_prob(raw_action)

        # Tanh correction
        log_prob -= torch.log(
            1 - action.pow(2) + 1e-6
        )

        log_prob = log_prob.sum(dim=-1)

        # ------------------------------------------------------------
        # Entropy
        # ------------------------------------------------------------

        entropy = dist.entropy().sum(dim=-1)

        # ------------------------------------------------------------
        # Critic
        # ------------------------------------------------------------

        value = self.critic(features).squeeze(-1)

        return (
            action,
            log_prob,
            entropy,
            value
        )