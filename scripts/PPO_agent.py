import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from PPO_network import ActorCriticNetwork

class PPOAgent:
    """
    PPO agent for continuous action spaces.

    Based on: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

    The main idea is to collect a rollout of experience, then do multiple
    epochs of gradient updates using a clipped objective that prevents the
    policy from changing too drastically in one step.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_steps: int = 2048,            # how many env steps to collect before each update
        n_epochs: int = 10,             # how many times to loop over the rollout data
        batch_size: int = 64,
        total_timesteps: int = 1_000_000,
        gamma: float = 0.99,            # discount factor
        gae_lambda: float = 0.95,       # GAE lambda - controls bias/variance tradeoff
        clip_eps: float = 0.2,          # PPO clipping range
        vf_coef: float = 0.5,           # how much to weight the value loss
        ent_coef: float = 0.05,         # entropy bonus weight - encourages exploration
        max_grad_norm: float = 0.5,     # gradient clipping
        lr: float = 3e-4,
        anneal_lr: bool = True,         # decay LR linearly to 0 over training
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.network = ActorCriticNetwork(obs_dim, action_dim).to(self.device)
        self.optimizer = Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Linearly decay the learning rate to ~0 over all updates
        n_updates = total_timesteps // n_steps
        self.scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=n_updates
        ) if anneal_lr else None

        self._init_storage()
        self._step = 0

    def _init_storage(self):
        """Pre-allocate buffers"""
        T, D, A = self.n_steps, self.obs_dim, self.action_dim
        self.obs_buffer = torch.zeros(T, D, device=self.device)
        self.action_buffer = torch.zeros(T, A, device=self.device)
        self.log_prob_buffer = torch.zeros(T, device=self.device)
        self.reward_buffer = torch.zeros(T, device=self.device)
        self.done_buffer = torch.zeros(T, device=self.device)
        self.value_buffer = torch.zeros(T, device=self.device)

        self.reward_mean = 0.0
        self.reward_var  = 1.0
        self.reward_std  = 1.0
        self.reward_count = 1e-4

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """
        Pick an action for the current observation and store the transition.
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action, log_prob, entropy, value = self.network.get_action_and_value(obs_t)

        self.obs_buffer[self._step] = obs_t.squeeze(0)
        self.action_buffer[self._step] = action.squeeze(0)
        self.log_prob_buffer[self._step] = log_prob.squeeze(0)
        self.value_buffer[self._step] = value.squeeze(0)
        self._step += 1

        return action.squeeze(0).cpu().numpy(), log_prob.item(), value.item()

    # def store_reward_done(self, step: int, reward: float, done: bool):
    #     """Store the reward and terminal flag after env.step()."""
    #     self.reward_buffer[step] = reward
    #     self.done_buffer[step] = float(done)

    def store_reward_done(self, step: int, reward: float, done: bool):
        """Store the reward and terminal flag after env.step()."""
        # Running reward normalization (Welford's algorithm)
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_var = self.reward_var + delta * delta2
        self.reward_std = max((self.reward_var / self.reward_count) ** 0.5, 1e-8)

        self.reward_buffer[step] = (reward - self.reward_mean) / self.reward_std
        self.done_buffer[step]   = float(done)

    @torch.no_grad()
    def compute_gae(self, next_obs: np.ndarray, next_done: bool):
        """
        Compute GAE advantages and discounted returns.

        https://medium.com/deepgamingai/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-22337981f815
        """
        next_obs_t = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
        next_value = self.network.get_value(next_obs_t).squeeze()
        next_done_t = torch.tensor(float(next_done), device=self.device)

        advantages = torch.zeros_like(self.reward_buffer)
        last_gae = torch.tensor(0.0, device=self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - next_done_t
            else:
                next_non_terminal = 1.0 - self.done_buffer[t + 1]
                next_value = self.value_buffer[t + 1]

            delta = self.reward_buffer[t] + self.gamma * next_value * next_non_terminal - self.value_buffer[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.value_buffer
        return advantages, returns

    def update(self, next_obs: np.ndarray, next_done: bool) -> dict:
        """
        Run the PPO update.

        For each epoch we shuffle the data into mini-batches and compute:
          - Policy loss: clipped surrogate objective
          - Value loss: MSE between predicted values and GAE returns
          - Entropy: bonus to prevent the policy collapsing too early
        """
        advantages, returns = self.compute_gae(next_obs, next_done)

        # Normalize advantages - reduces variance and stabilizes training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_batches = 0

        indices = np.arange(self.n_steps)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, self.n_steps, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]

                action, new_log_prob, entropy, new_value = \
                    self.network.get_action_and_value(
                        self.obs_buffer[mb_idx],
                        self.action_buffer[mb_idx]
                    )

                # Ratio between new and old policy probabilities
                ratio = (new_log_prob - self.log_prob_buffer[mb_idx]).exp()

                mb_adv = advantages[mb_idx]

                # Clipped policy loss
                loss_unclipped = ratio * mb_adv
                loss_clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(loss_unclipped, loss_clipped).mean()

                # Value loss
                new_value  = new_value.squeeze()
                value_loss = 0.5 * (new_value - returns[mb_idx]).pow(2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_loss += loss.item()
                n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        self._step = 0

        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "total_loss": total_loss / n_batches,
        }

    def save(self, path: str):
        torch.save({"network": self.network.state_dict(), "optimizer": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])