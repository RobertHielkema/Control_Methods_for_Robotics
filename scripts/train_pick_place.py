import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import deque

import numpy as np
import torch

from PPO_agent import PPOAgent
from pick_place_env import PickPlaceEnv, OBS_DIM, EPISODE_STEPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--start-phase", type=int, default=0, choices=[0, 1, 2],
                        help="0: full task, 1: start above cube, 2: start grasped")
    parser.add_argument("--save-dir", default="checkpoints_pick_place")
    parser.add_argument("--resume", default=None, help="checkpoint to continue training from")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = PickPlaceEnv(start_phase=args.start_phase, realtime=False)

    n_steps = EPISODE_STEPS * 10
    agent = PPOAgent(
        obs_dim=OBS_DIM,
        action_dim=6,
        n_steps=n_steps,
        n_epochs=10,
        batch_size=64,
        total_timesteps=2_000_000,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=3e-4,
        anneal_lr=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")

    log_path = os.path.join(args.save_dir, "episode_log.csv")
    log_file = open(log_path, "a" if args.resume else "w")
    if not args.resume:
        log_file.write("episode,start_phase,episode_return,steps,max_phase,lifted,dropped,knocked,placed,final_dist\n")

    recent_returns = deque(maxlen=20)
    best_mean_return = -np.inf

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            episode_return = 0.0
            max_phase = env.phase
            lifted = False

            for t in range(EPISODE_STEPS):
                action, _, _ = agent.select_action(obs)
                obs, reward, done, info = env.step(action)

                agent.store_reward_done(agent._step - 1, reward, done)

                episode_return += reward
                max_phase = max(max_phase, info["phase"])
                lifted = lifted or info["held"]

                if agent._step == agent.n_steps:
                    upd = agent.update(obs, done)
                    print(
                        f"UPDATE | Policy loss: {upd['policy_loss']:.4f} | "
                        f"Value loss: {upd['value_loss']:.4f} | "
                        f"Entropy: {upd['entropy']:.4f} | "
                        f"KL: {upd['approx_kl']:.5f} | "
                        f"Clip frac: {upd['clip_fraction']:.3f}"
                    )

                if done:
                    break

            print(
                f"Episode {ep} | return: {episode_return:.1f} | steps: {t + 1} | "
                f"phase reached: {max_phase} | lifted: {int(lifted)} | dropped: {int(info['dropped'])} | "
                f"knocked: {int(info['knocked'])} | placed: {int(info['placed'])} | "
                f"final dist: {info['dist']:.3f}"
            )
            log_file.write(
                f"{ep},{args.start_phase},{episode_return:.2f},{t + 1},{max_phase},{int(lifted)},"
                f"{int(info['dropped'])},{int(info['knocked'])},{int(info['placed'])},{info['dist']:.4f}\n"
            )
            log_file.flush()

            recent_returns.append(episode_return)
            if len(recent_returns) == recent_returns.maxlen:
                mean_return = float(np.mean(recent_returns))
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    agent.save(os.path.join(args.save_dir, "best.pt"))
            if (ep + 1) % 50 == 0:
                agent.save(os.path.join(args.save_dir, "last.pt"))
    finally:
        agent.save(os.path.join(args.save_dir, "last.pt"))
        log_file.close()
        print(f"Saved checkpoints to {args.save_dir}/ (best.pt, last.pt)")


if __name__ == "__main__":
    main()
