import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
from numpy import random


def layer_init(layer, std=np.sqrt(2.0), bias_const=0.0):
    # Orthogonal init is recommended for PPO - helps with stable early training
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO with continuous action spaces.

    Shared trunk feeds into two heads:
      - Actor: outputs the mean of a Gaussian policy
      - Critic: outputs a scalar value estimate V(s)
    """

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        # --- Shared --
        self.shared_fc1  = layer_init(nn.Linear(obs_dim, 256))
        self.shared_act1 = nn.Tanh()
        self.shared_fc2  = layer_init(nn.Linear(256, 256))
        self.shared_act2 = nn.Tanh()


        # --- Actor head ---
        self.actor_fc1   = layer_init(nn.Linear(256, 256))
        self.actor_act1  = nn.Tanh()
        self.actor_mean  = layer_init(nn.Linear(256, action_dim), std=0.01)
        initial_std = 0.5
        self.actor_log_std = nn.Parameter(
            torch.full((1, action_dim), np.log(initial_std))
        )

        # -- Critic head --
        self.critic_fc1  = layer_init(nn.Linear(256, 256))
        self.critic_act1 = nn.Tanh()
        self.critic_out  = layer_init(nn.Linear(256, 1), std=1.0)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.shared_fc1(obs)
        x = self.shared_act1(x)
        x = self.shared_fc2(x)
        x = self.shared_act2(x)
        x = self.critic_fc1(x)
        x = self.critic_act1(x)
        return self.critic_out(x).squeeze(-1)

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor = None, nr_current_ep = 1,  total_episode_count = 10):
        # Shared
        x = self.shared_fc1(obs)
        x = self.shared_act1(x)
        x = self.shared_fc2(x)
        x = self.shared_act2(x)

        # Actor
        a = self.actor_fc1(x)
        a = self.actor_act1(a)
        action_mean = self.actor_mean(a)
        action_std  = self.actor_log_std.exp().expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)

        # Critic
        c     = self.critic_fc1(x)
        c     = self.critic_act1(c)
        value = self.critic_out(c).squeeze(-1)

        return action, log_prob, entropy, value