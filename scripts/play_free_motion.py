import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import numpy as np
import pybullet as p

from PPO_agent import PPOAgent
from classic_controller_gripper_PPO_fixed import (
    create_scene,
    create_observation,
    get_ee_position_fk,
    reset_episode,
    ACTION_REPEAT,
    ACTION_SCALE,
    EPISODE_STEPS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints_free/best.pt")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--torque-log", default="free_motion_torques.csv")
    parser.add_argument("--every", type=int, default=1)
    args = parser.parse_args()

    robot, plane_id = create_scene()
    robot_id = robot.robot_id
    robot.control_finger_width(0.1)

    q_min, q_max = [], []
    for idx in robot.arm_indices:
        info = p.getJointInfo(robot_id, int(idx))
        q_min.append(info[8])
        q_max.append(info[9])
    q_min, q_max = np.array(q_min), np.array(q_max)

    agent = PPOAgent(obs_dim=18, action_dim=6, anneal_lr=False, device="cpu")
    agent.load(args.checkpoint)

    Kp = np.diag(np.full(6, 50.0))
    Kd = np.diag(np.full(6, 20.0))

    log_file = open(args.torque_log, "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["episode", "time_s"] + [f"tau_{j}" for j in range(6)])
    physics_step = 0
    max_abs_torque = np.zeros(6)

    for ep in range(args.episodes):
        physics_step = 0
        target = reset_episode(robot, q_min, q_max)
        p.addUserDebugPoints([target.tolist()], [[0, 0, 1]], pointSize=10)

        q = robot.robot_state.get_joint_angles()
        q_dot = robot.robot_state.get_joint_velocities()
        ee = get_ee_position_fk(robot)
        obs = create_observation(q, q_dot, ee, target)

        for t in range(EPISODE_STEPS):
            action = agent.act_deterministic(obs)

            q = robot.robot_state.get_joint_angles()
            q_target = np.clip(q + ACTION_SCALE * np.clip(action, -1.0, 1.0), q_min, q_max)

            for _ in range(ACTION_REPEAT):
                dyn = robot.robot_model.get_dynamics()
                q = robot.robot_state.get_joint_angles()
                q_dot = robot.robot_state.get_joint_velocities()
                accel = Kp @ (q_target - q) - Kd @ q_dot
                torques = dyn.mass_matrix @ accel + dyn.coriolis_vector + dyn.gravity_vector
                applied = np.clip(torques, -robot.arm_joint_forces, robot.arm_joint_forces)
                robot.control_torques(applied)

                if physics_step % args.every == 0:
                    writer.writerow([ep, f"{physics_step * robot.sim.timestep:.4f}"] + [f"{x:.4f}" for x in applied])
                max_abs_torque = np.maximum(max_abs_torque, np.abs(applied))
                physics_step += 1

                robot.sim.step(realtime=True)

            q = robot.robot_state.get_joint_angles()
            q_dot = robot.robot_state.get_joint_velocities()
            ee = robot.robot_state.get_end_effector_position()
            obs = create_observation(q, q_dot, ee, target)

        print(f"Episode {ep} | final dist: {np.linalg.norm(target - ee):.3f}")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    main()