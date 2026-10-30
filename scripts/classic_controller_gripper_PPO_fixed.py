import sys, os

import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from PPO_agent import PPOAgent

# --------------------------------------------------
# CHANGED: new constants for the control setup
# --------------------------------------------------
ACTION_REPEAT = 20       # physics steps per policy step (1 kHz sim -> 50 Hz policy)
ACTION_SCALE = 0.1       # max joint change (rad) the policy can request per policy step
EPISODE_STEPS = 1000      # policy steps per episode (150 * 20 ms = 3 s)
SUCCESS_DIST = 0.05      # meters, only used for logging now
PROGRESS_WEIGHT = 10.0   # weight of the progress reward term

# CHANGED: set to e.g. np.array([-0.7, 0.5, 0.5]) for the fixed-target sanity run.
# Keep None for random reachable targets. Check that the fixed target is reachable!
FIXED_TARGET = np.array([-0.7, 0.5, 0.5])

Q_NEUTRAL = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])


def create_scene():
    robot = UR7e(
        block_gripper=False,
        neutral_joints=Q_NEUTRAL,
    )
    plane_id = p.loadURDF("plane.urdf")
    return robot, plane_id


# --------------------------------------------------
# CHANGED: observation now includes q_dot and the relative position error,
# and everything is scaled to roughly [-1, 1]
# --------------------------------------------------
def create_observation(q, q_dot, ee_position, target):
    return np.concatenate([
        q / np.pi,
        q_dot * 0.1,
        ee_position,
        target - ee_position,
    ]).astype(np.float32)


# --------------------------------------------------
# CHANGED: reward = your original log term + progress term (replaces cosine similarity)
# --------------------------------------------------
def calculate_reward(dist, prev_dist):
    w_1 = 0.001
    t_e = 0.0001
    reward = -(w_1 * dist ** 2 + np.log(dist ** 2 + t_e))
    reward += PROGRESS_WEIGHT * (prev_dist - dist)
    return reward


# --------------------------------------------------
# CHANGED: forward kinematics recomputed, so the value is correct right after a reset
# --------------------------------------------------
def get_ee_position_fk(robot):
    state = p.getLinkState(
        robot.robot_id,
        robot.robot_state.ee_index,
        computeForwardKinematics=True,
    )
    return np.array(state[4])


# --------------------------------------------------
# CHANGED: reachable target = end-effector position of a random configuration,
# and the start pose is a small perturbation of the neutral pose
# --------------------------------------------------
def reset_episode(robot, q_min, q_max):
    if FIXED_TARGET is not None:
        target = FIXED_TARGET.copy()
        robot.set_joint_angles(Q_NEUTRAL)
        return target

    while True:
        q_rand = np.clip(
            np.random.uniform(Q_NEUTRAL - 1.0, Q_NEUTRAL + 1.0), q_min, q_max
        )
        robot.set_joint_angles(q_rand)
        target = get_ee_position_fk(robot)
        if target[2] > 0.1:  # keep targets above the floor
            break

    q_start = np.clip(Q_NEUTRAL + np.random.uniform(-0.3, 0.3, 6), q_min, q_max)
    robot.set_joint_angles(q_start)
    return target


def run():

    # --------------------------------------------------
    # 1. Create environment
    # --------------------------------------------------
    robot, plane_id = create_scene()
    robot_id = robot.robot_id
    robot.control_finger_width(0.1)

    # CHANGED: joint limits taken from the arm joints only, and actually used now
    q_min, q_max = [], []
    for idx in robot.arm_indices:
        info = p.getJointInfo(robot_id, int(idx))
        q_min.append(info[8])
        q_max.append(info[9])
    q_min = np.array(q_min)
    q_max = np.array(q_max)
    print("Joint limits:", q_min, q_max)

    # --------------------------------------------------
    # 2. Get initial observation (only needed to get obs_dim)
    # --------------------------------------------------
    target = reset_episode(robot, q_min, q_max)
    q = robot.robot_state.get_joint_angles()
    q_dot = robot.robot_state.get_joint_velocities()
    ee = get_ee_position_fk(robot)
    observation = create_observation(q, q_dot, ee, target)

    # --------------------------------------------------
    # 3. Create PPO agent
    # --------------------------------------------------
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
        anneal_lr=False,         # CHANGED: off while debugging
        device="cuda" if torch.cuda.is_available() else "cpu",  # CHANGED: fallback
    )

    # --------------------------------------------------
    # 4. Controller gains
    # --------------------------------------------------
    # CHANGED: Kp=5/Kd=2 was too slow (settling time ~4 s vs a 2 s episode).
    # omega_n ~ 14 rad/s, zeta ~ 1.4 (a bit overdamped, which is safe)
    n_joints = 6
    Kp = np.diag(np.full(n_joints, 50.0))
    Kd = np.diag(np.full(n_joints, 20.0))

    # CHANGED: simple CSV log, useful for learning curves in your report
    log_file = open("episode_log.csv", "w")
    log_file.write("episode,final_dist,min_dist,episode_return,success\n")

    # --------------------------------------------------
    # 5. Main training loop
    # --------------------------------------------------
    num_episodes = 1000
    debugpoint = None

    for ep in range(num_episodes):

        # CHANGED: reset, then rebuild the observation from the NEW state
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

            # ----------------------------------------------
            # A. Policy picks an action
            # ----------------------------------------------
            action, log_prob, value = agent.select_action(observation)

            # CHANGED: action = bounded JOINT DELTA, not an absolute joint target
            q = robot.robot_state.get_joint_angles()
            q_target = np.clip(q + ACTION_SCALE * np.clip(action, -1.0, 1.0), q_min, q_max)

            # ----------------------------------------------
            # B. CHANGED: action repeat, inverse dynamics controller tracks q_target
            # ----------------------------------------------
            for _ in range(ACTION_REPEAT):
                dynamics = robot.robot_model.get_dynamics()
                M = dynamics.mass_matrix
                C = dynamics.coriolis_vector
                g = dynamics.gravity_vector

                q = robot.robot_state.get_joint_angles()
                q_dot = robot.robot_state.get_joint_velocities()

                desired_accel = Kp @ (q_target - q) - Kd @ q_dot
                torques = M @ desired_accel + C + g

                # control_torques already clips to the motor limits
                robot.control_torques(torques)

                # CHANGED: realtime=False, otherwise it sleeps every step
                robot.sim.step(realtime=False)

            # ----------------------------------------------
            # C. Read NEW state after physics
            # ----------------------------------------------
            q = robot.robot_state.get_joint_angles()
            q_dot = robot.robot_state.get_joint_velocities()
            ee = robot.robot_state.get_end_effector_position()
            dist = np.linalg.norm(target - ee)

            # ----------------------------------------------
            # D. Reward
            # ----------------------------------------------
            reward = calculate_reward(dist, prev_dist)
            prev_dist = dist
            min_dist = min(min_dist, dist)
            episode_return += reward

            # ----------------------------------------------
            # E. CHANGED: done only marks the end of the episode (time limit).
            # Reaching the target no longer sets done, the agent just keeps
            # collecting reward by staying close.
            # ----------------------------------------------
            done = (t == EPISODE_STEPS - 1)

            buffer_index = agent._step - 1
            agent.store_reward_done(buffer_index, reward, done)

            # ----------------------------------------------
            # F. Next observation
            # ----------------------------------------------
            observation = create_observation(q, q_dot, ee, target)

            # ----------------------------------------------
            # G. PPO update after collecting n_steps
            # ----------------------------------------------
            if agent._step == agent.n_steps:
                info = agent.update(observation, done)
                print(
                    f"UPDATE | Policy loss: {info['policy_loss']:.4f} | "
                    f"Value loss: {info['value_loss']:.4f} | "
                    f"Entropy: {info['entropy']:.4f} | "
                    f"KL: {info['approx_kl']:.5f} | "
                    f"Clip frac: {info['clip_fraction']:.3f}"
                )

        # CHANGED: per-episode summary instead of printing every 200 steps
        success = int(dist < SUCCESS_DIST)
        print(
            f"Episode {ep} | final dist: {dist:.3f} | min dist: {min_dist:.3f} | "
            f"return: {episode_return:.1f} | success: {success}"
        )
        log_file.write(f"{ep},{dist:.4f},{min_dist:.4f},{episode_return:.2f},{success}\n")
        log_file.flush()

        agent.current_ep += 1

    log_file.close()


if __name__ == "__main__":
    run()
