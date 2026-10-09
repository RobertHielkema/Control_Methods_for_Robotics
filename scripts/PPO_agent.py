import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

from PPO_network import ActorCriticNetwork


class PPOAgent:
    """
    PPO agent for continuous action spaces.

    Collects n_steps transitions, computes GAE advantages,
    and performs PPO clipped-objective updates.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        total_timesteps: int = 1_000_000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        lr: float = 3e-4,
        anneal_lr: bool = True,
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

        self.network = ActorCriticNetwork(
            obs_dim, action_dim
        ).to(self.device)

        self.optimizer = Adam(
            self.network.parameters(),
            lr=lr,
            eps=1e-5,
        )

        n_updates = max(1, total_timesteps // n_steps)

        self.scheduler = (
            LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1e-8,
                total_iters=n_updates,
            )
            if anneal_lr
            else None
        )

        self._init_storage()
        self._step = 0

    def _init_storage(self):
        T, D, A = self.n_steps, self.obs_dim, self.action_dim

        self.obs_buffer = torch.zeros(
            T, D, device=self.device
        )
        self.action_buffer = torch.zeros(
            T, A, device=self.device
        )
        self.log_prob_buffer = torch.zeros(
            T, device=self.device
        )
        self.reward_buffer = torch.zeros(
            T, device=self.device
        )
        self.done_buffer = torch.zeros(
            T, device=self.device
        )
        self.value_buffer = torch.zeros(
            T, device=self.device
        )

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        """
        Select an action and store its observation, log probability,
        and critic value. Store the reward/done after env.step().
        """
        if self._step >= self.n_steps:
            raise RuntimeError(
                "Rollout buffer is full. Call agent.update() first."
            )

        obs_t = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        action, log_prob, _, value = (
            self.network.get_action_and_value(obs_t)
        )

        idx = self._step
        self.obs_buffer[idx] = obs_t.squeeze(0)
        self.action_buffer[idx] = action.squeeze(0)
        self.log_prob_buffer[idx] = log_prob.reshape(())
        self.value_buffer[idx] = value.reshape(())

        self._step += 1

        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def store_reward_done(
        self, step: int, reward: float, done: bool
    ):
        """
        Store the raw reward and whether this transition ended
        the episode. `step` is the rollout-buffer index.
        """
        if not 0 <= step < self.n_steps:
            raise IndexError("Invalid rollout-buffer index.")

        self.reward_buffer[step] = float(reward)
        self.done_buffer[step] = float(done)

    @torch.no_grad()
    def compute_gae(
        self, next_obs: np.ndarray, next_done: bool
    ):
        """
        Compute GAE advantages and returns.

        `next_obs` is the observation after the last collected
        transition. `next_done` indicates whether that transition
        ended the episode.
        """
        if self._step != self.n_steps:
            raise RuntimeError(
                f"Expected {self.n_steps} transitions, "
                f"but only {self._step} were collected."
            )

        next_obs_t = torch.as_tensor(
            next_obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        next_value = self.network.get_value(
            next_obs_t
        ).reshape(())

        advantages = torch.zeros_like(self.reward_buffer)
        last_gae = torch.zeros((), device=self.device)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                # Bootstrap only if the final transition did not end
                # the episode.
                next_non_terminal = 1.0 - float(next_done)
                next_val = next_value
            else:
                # done_buffer[t] belongs to transition t.
                next_non_terminal = 1.0 - self.done_buffer[t]
                next_val = self.value_buffer[t + 1]

            delta = (
                self.reward_buffer[t]
                + self.gamma * next_val * next_non_terminal
                - self.value_buffer[t]
            )

            last_gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_non_terminal
                * last_gae
            )

            advantages[t] = last_gae

        returns = advantages + self.value_buffer
        return advantages, returns

    def update(
        self, next_obs: np.ndarray, next_done: bool
    ) -> dict:
        """
        Perform PPO updates on a full rollout.
        """
        advantages, returns = self.compute_gae(
            next_obs, next_done
        )

        # Preserve unnormalized advantages for diagnostics.
        raw_advantages = advantages.clone()

        # Normalize advantages for the policy objective.
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_clip_fraction = 0.0
        n_batches = 0

        indices = np.arange(self.n_steps)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(
                0, self.n_steps, self.batch_size
            ):
                mb_idx_np = indices[
                    start:start + self.batch_size
                ]

                mb_idx = torch.as_tensor(
                    mb_idx_np,
                    dtype=torch.long,
                    device=self.device,
                )

                _, new_log_prob, entropy, new_value = (
                    self.network.get_action_and_value(
                        self.obs_buffer[mb_idx],
                        self.action_buffer[mb_idx],
                    )
                )

                new_log_prob = new_log_prob.reshape(-1)
                new_value = new_value.reshape(-1)
                mb_adv = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                old_log_prob = self.log_prob_buffer[mb_idx]

                log_ratio = new_log_prob - old_log_prob
                ratio = log_ratio.exp()

                loss_unclipped = ratio * mb_adv
                loss_clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    )
                    * mb_adv
                )

                policy_loss = -torch.minimum(
                    loss_unclipped, loss_clipped
                ).mean()

                value_loss = 0.5 * (
                    new_value - mb_returns
                ).pow(2).mean()

                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_eps
                    ).float().mean()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        metrics = {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
            "approx_kl": total_approx_kl / n_batches,
            "clip_fraction": total_clip_fraction / n_batches,
            "reward_mean": self.reward_buffer.mean().item(),
            "reward_std": self.reward_buffer.std(
                unbiased=False
            ).item(),
            "return_mean": returns.mean().item(),
            "return_std": returns.std(
                unbiased=False
            ).item(),
            "value_mean": self.value_buffer.mean().item(),
            "advantage_mean": raw_advantages.mean().item(),
            "advantage_std": raw_advantages.std(
                unbiased=False
            ).item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        # Rollout has been consumed; prepare for the next rollout.
        self._step = 0

        return metrics

    def save(self, path: str):
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(
            path, map_location=self.device
        )

        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if (
            self.scheduler is not None
            and checkpoint.get("scheduler") is not None
        ):
            self.scheduler.load_state_dict(
                checkpoint["scheduler"]
            )