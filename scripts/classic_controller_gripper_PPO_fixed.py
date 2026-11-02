
import sys, os
 
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 
import argparse
from collections import deque
 
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from PPO_agent import PPOAgent
 
ACTION_REPEAT = 20       # physics steps per policy step
ACTION_SCALE = 0.1       # max joint change per policy step
EPISODE_STEPS = 1000
SUCCESS_DIST = 0.05
PROGRESS_WEIGHT = 10.0
 
FIXED_TARGET = np.array([-0.7, 0.5, 0.5])
# FIXED_TARGET = None
 
Q_NEUTRAL = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
 
 
def create_scene():
    robot = UR7e(
        block_gripper=False,
        neutral_joints=Q_NEUTRAL,
    )
    plane_id = p.loadURDF("plane.urdf")
    return robot, plane_id
 
 
# Observation contains: q, q_dot, ee_position, error to the target
def create_observation(q, q_dot, ee_position, target):
    return np.concatenate([
        q / np.pi,
        q_dot * 0.1,
        ee_position,
        target - ee_position,
    ]).astype(np.float32)
 
 
def calculate_reward(dist, prev_dist):
    w_1 = 0.001
    t_e = 0.0001
    reward = -(w_1 * dist ** 2 + np.log(dist ** 2 + t_e))
    reward += PROGRESS_WEIGHT * (prev_dist - dist)
    return reward
 
 
def get_ee_position_fk(robot):
    state = p.getLinkState(
        robot.robot_id,
        robot.robot_state.ee_index,
        computeForwardKinematics=True,
    )
    return np.array(state[4])
 
 
def reset_episode(robot, q_min, q_max):
    if FIXED_TARGET is not None:
        target = FIXED_TARGET.copy()
        robot.set_joint_angles(Q_NEUTRAL)
        return target
 
    # reachable target = ee position of a random joint configuration
    while True:
        q_rand = np.clip(
            np.random.uniform(Q_NEUTRAL - 1.0, Q_NEUTRAL + 1.0), q_min, q_max
        )
        robot.set_joint_angles(q_rand)
        target = get_ee_position_fk(robot)
        if target[2] > 0.1:
            break
 
    q_start = np.clip(Q_NEUTRAL + np.random.uniform(-0.3, 0.3, 6), q_min, q_max)
    robot.set_joint_angles(q_start)
    return target
 
 
def run(args):
 
    robot, plane_id = create_scene()
    robot_id = robot.robot_id
    robot.control_finger_width(0.1)
 
    q_min, q_max = [], []
    for idx in robot.arm_indices:
        info = p.getJointInfo(robot_id, int(idx))
        q_min.append(info[8])
        q_max.append(info[9])
    q_min = np.array(q_min)
    q_max = np.array(q_max)
    print("Joint limits:", q_min, q_max)
 
    # initial observation, only needed for obs_dim
    target = reset_episode(robot, q_min, q_max)
    q = robot.robot_state.get_joint_angles()
    q_dot = robot.robot_state.get_joint_velocities()
    ee = get_ee_position_fk(robot)
    observation = create_observation(q, q_dot, ee, target)
 
    agent = PPOAgent(
        obs_dim=observation.shape[0],
        action_dim=6,
        n_steps=8000,
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
 
    os.makedirs(args.save_dir, exist_ok=True)
    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")
 
    n_joints = 6
    Kp = np.diag(np.full(n_joints, 50.0))
    Kd = np.diag(np.full(n_joints, 20.0))
 
    log_file = open(os.path.join(args.save_dir, "episode_log.csv"), "a" if args.resume else "w")
    if not args.resume:
        log_file.write("episode,final_dist,min_dist,episode_return,success\n")
 
    num_episodes = args.episodes
    debugpoint = None
    recent_returns = deque(maxlen=20)
    best_mean_return = -np.inf
 
    try:
        for ep in range(num_episodes):
 
            # new target and reset
            target = reset_episode(robot, q_min, q_max)
            if debugpoint is not None:
                p.removeUserDebugItem(debugpoint)
            debugpoint = p.addUserDebugPoints([target.tolist()], [[0, 0, 1]], pointSize=10)
 
            q = robot.robot_state.get_joint_angles()
            q_dot = robot.robot_state.get_joint_velocities()
            ee = get_ee_position_fk(robot)
            observation = create_observation(q, q_dot, ee, target)
            prev_dist = np.linalg.norm(target - ee)
 
            min_dist = prev_dist
            episode_return = 0.0
 
            print(f"Starting Episode Number:{ep}/{num_episodes} | target: {np.round(target, 3)}")
 
            for t in range(EPISODE_STEPS):
 
                # action is a joint delta
                action, log_prob, value = agent.select_action(observation)
 
                q = robot.robot_state.get_joint_angles()
                q_target = np.clip(q + ACTION_SCALE * np.clip(action, -1.0, 1.0), q_min, q_max)
 
                # inverse dynamics controller tracks q_target
                for _ in range(ACTION_REPEAT):
                    dynamics = robot.robot_model.get_dynamics()
                    M = dynamics.mass_matrix
                    C = dynamics.coriolis_vector
                    g = dynamics.gravity_vector
 
                    q = robot.robot_state.get_joint_angles()
                    q_dot = robot.robot_state.get_joint_velocities()
 
                    desired_accel = Kp @ (q_target - q) - Kd @ q_dot
                    torques = M @ desired_accel + C + g
 
                    robot.control_torques(torques)
                    robot.sim.step(realtime=False)
 
                q = robot.robot_state.get_joint_angles()
                q_dot = robot.robot_state.get_joint_velocities()
                ee = robot.robot_state.get_end_effector_position()
                dist = np.linalg.norm(target - ee)
 
                reward = calculate_reward(dist, prev_dist)
                prev_dist = dist
                min_dist = min(min_dist, dist)
                episode_return += reward
 
                # only the time limit ends an episode
                done = (t == EPISODE_STEPS - 1)
 
                buffer_index = agent._step - 1
                agent.store_reward_done(buffer_index, reward, done)
 
                observation = create_observation(q, q_dot, ee, target)
 
                if agent._step == agent.n_steps:
                    info = agent.update(observation, done)
                    print(
                        f"UPDATE | Policy loss: {info['policy_loss']:.4f} | "
                        f"Value loss: {info['value_loss']:.4f} | "
                        f"Entropy: {info['entropy']:.4f} | "
                        f"KL: {info['approx_kl']:.5f} | "
                        f"Clip frac: {info['clip_fraction']:.3f}"
                    )
 
            success = int(dist < SUCCESS_DIST)
            print(
                f"Episode {ep} | final dist: {dist:.3f} | min dist: {min_dist:.3f} | "
                f"return: {episode_return:.1f} | success: {success}"
            )
            log_file.write(f"{ep},{dist:.4f},{min_dist:.4f},{episode_return:.2f},{success}\n")
            log_file.flush()
 
            # save the best model (rolling mean return) and the last one
            recent_returns.append(episode_return)
            if len(recent_returns) == recent_returns.maxlen:
                mean_return = float(np.mean(recent_returns))
                if mean_return > best_mean_return:
                    best_mean_return = mean_return
                    agent.save(os.path.join(args.save_dir, "best.pt"))
            if (ep + 1) % 50 == 0:
                agent.save(os.path.join(args.save_dir, "last.pt"))
 
            agent.current_ep += 1
 
    finally:
        agent.save(os.path.join(args.save_dir, "last.pt"))
        log_file.close()
        print(f"Saved checkpoints to {args.save_dir}/ (best.pt, last.pt)")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--save-dir", default="checkpoints_free")
    parser.add_argument("--resume", default=None, help="checkpoint to continue training from")
    run(parser.parse_args())
