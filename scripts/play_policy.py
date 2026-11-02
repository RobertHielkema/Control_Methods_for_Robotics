import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from PPO_agent import PPOAgent
from button_press_env import ButtonPressEnv, OBS_DIM, EPISODE_STEPS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_button_final/best.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--full-motion", action="store_false")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    env = ButtonPressEnv(start_at_waypoint_1=not args.full_motion, realtime=True)

    agent = PPOAgent(obs_dim=OBS_DIM, action_dim=6, anneal_lr=False, device="cpu")
    agent.load(args.checkpoint)

    log_file = None
    if args.log:
        log_file = open(args.log, "w")
        log_file.write("episode,time_s,phase,dist,force,sensor_fz\n")

    for ep in range(args.episodes):
        obs = env.reset()
        max_force = 0.0
        for t in range(EPISODE_STEPS):
            action = agent.act_deterministic(obs)

            obs, reward, done, info = env.step(action)
            max_force = max(max_force, info["force"])
            if log_file:
                time_s = (t + 1) * 0.02
                log_file.write(
                    f"{ep},{time_s:.2f},{info['phase']},{info['dist']:.4f},"
                    f"{info['force']:.2f},{info['sensor_fz']:.2f}\n"
                )
        print(f"Episode {ep} | final dist: {info['dist']:.3f} | max force: {max_force:.1f} N | phase: {info['phase']}")

    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()