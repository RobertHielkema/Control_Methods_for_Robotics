import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from collections import deque

import numpy as np
import torch

from PPO_agent import PPOAgent
from button_press_env import ButtonPressEnv, OBS_DIM, EPISODE_STEPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--full-motion", action="store_true",
                        help="start near the neutral pose (reach the point above the button first). "
                             "Default: start above the button, press only.")
    parser.add_argument("--save-dir", default="checkpoints_button")
    parser.add_argument("--resume", default=None, help="checkpoint to continue training from")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    env = ButtonPressEnv(start_at_waypoint_1=not args.full_motion, realtime=False)

    n_steps = EPISODE_STEPS * 10          # 10 whole episodes per PPO update
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
        log_file.write("episode,episode_return,final_dist,min_dist,max_force,press_steps,phase_reached,press_success\n")

    recent_returns = deque(maxlen=20)
    best_mean_return = -np.inf

    try:
        for ep in range(args.episodes):
            obs = env.reset()
            episode_return = 0.0
            max_force = 0.0
            press_steps = 0
            min_dist = np.inf

            for t in range(EPISODE_STEPS):
                action, _, _ = agent.select_action(obs)
                obs, reward, done, info = env.step(action)

                agent.store_reward_done(agent._step - 1, reward, done)

                episode_return += reward
                max_force = max(max_force, info["force"])
                min_dist = min(min_dist, info["dist"])
                press_steps += int(info["press_success"])

                if agent._step == agent.n_steps:
                    upd = agent.update(obs, done)
                    print(
                        f"UPDATE | Policy loss: {upd['policy_loss']:.4f} | "
                        f"Value loss: {upd['value_loss']:.4f} | "
                        f"Entropy: {upd['entropy']:.4f} | "
                        f"KL: {upd['approx_kl']:.5f} | "
                        f"Clip frac: {upd['clip_fraction']:.3f}"
                    )

            success = int(press_steps > 0)
            print(
                f"Episode {ep} | return: {episode_return:.1f} | final dist: {info['dist']:.3f} | "
                f"max force: {max_force:.1f} N | steps pressing >= target: {press_steps} | "
                f"phase: {info['phase']} | success: {success}"
            )
            log_file.write(
                f"{ep},{episode_return:.2f},{info['dist']:.4f},{min_dist:.4f},"
                f"{max_force:.2f},{press_steps},{info['phase']},{success}\n"
            )
            log_file.flush()

            # ---- checkpointing ----
            recent_returns.append(episode_return)
            if len(recent_returns) == recent_returns.maxlen:
                mean_return = float(np.mean(recent_returns))
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    agent.save(os.path.join(args.save_dir, "best.pt"))
            if (ep + 1) % 50 == 0:
                agent.save(os.path.join(args.save_dir, "last.pt"))
    finally:
        # also runs on Ctrl-C, so stopping early still keeps the model
        agent.save(os.path.join(args.save_dir, "last.pt"))
        log_file.close()
        print(f"Saved checkpoints to {args.save_dir}/ (best.pt, last.pt)")


if __name__ == "__main__":
    main()
