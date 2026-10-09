import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType
from XboxController import XboxController
from numpy.linalg import norm
from math import sin, cos
from PPO_agent import PPOAgent


def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=False,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)

    # # Spawn tray or smt idk
    # tray_id = p.loadURDF("tray/traybox.urdf", basePosition=[-0.7, 0.5, 0])
    #
    # # add a cube to simulation
    # cube_size = 0.01
    #
    # collision_shape = p.createCollisionShape(
    #     p.GEOM_BOX,
    #     halfExtents=[cube_size, cube_size, cube_size]
    # )
    #
    # visual_shape = p.createVisualShape(
    #     p.GEOM_BOX,
    #     halfExtents=[cube_size, cube_size, cube_size]
    # )
    #
    # cube_id = p.createMultiBody(
    #     baseMass=0.2,
    #     baseCollisionShapeIndex=collision_shape,
    #     baseVisualShapeIndex=visual_shape,
    #     basePosition=[0.7, 0.5, 0.1]
    # )

    return robot # cube_id, tray_id

# Observation contains: q, q_dot, ee_position, desired_position
def create_observation(q, q_dot, ee_position, desired_position):
    observation = np.concatenate([q, q_dot, ee_position, desired_position])
    return observation



# Unlike in gymnasium we dont get the luxury of a pre-made reward function
def calculate_reward(ee_position, desired_position):

    # I implemented the Reward function from one of Hamidreza's papers
    # https://arxiv.org/pdf/2210.00803

    w_1 = 0.001
    w_2 = 0.1
    d_max = 0.05
    t_e = 0.0001
    e =  np.linalg.norm(ee_position - desired_position)
    #v = []
    reward = -(w_1*e**2 + np.log(e**2 + t_e))

    return reward



def run():

    # --------------------------------------------------
    # 1. Create environment
    # --------------------------------------------------
    robot = create_scene()

    robot.control_finger_width(0.1)

    desired_end_effector_pos = np.array([-0.7, 0.5, 0.5])

    # --------------------------------------------------
    # 2. Get initial observation
    # --------------------------------------------------
    q = robot.robot_state.get_joint_angles()
    q_dot = robot.robot_state.get_joint_velocities()
    ee_location = robot.robot_state.get_end_effector_position()

    observation = create_observation(
        q,
        q_dot,
        ee_location,
        desired_end_effector_pos
    )

    # --------------------------------------------------
    # 3. Create PPO agent
    # --------------------------------------------------
    agent = PPOAgent(
        obs_dim=observation.shape[0],
        action_dim=6,
        n_steps=2048,
        n_epochs=10,
        batch_size=64,
        total_timesteps=1_000_000,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        lr=3e-4,
        anneal_lr=True,
        device="cpu"
    )

    # --------------------------------------------------
    # 4. Main training loop
    # --------------------------------------------------
    total_timesteps = 1_000_000
    q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0])
    q_dot_desired = np.zeros_like(q_desired)  # static setpoint -> zero desired velocity
    q_ddot_desired = np.zeros_like(q_desired)
    n_joints = 6
    Kp = np.diag(np.full(n_joints, 200))
    Kd = np.diag(np.full(n_joints, 45))
    for step in range(total_timesteps):
        dynamics = robot.robot_model.get_dynamics()
        M = dynamics.mass_matrix
        C = dynamics.coriolis_vector
        g = dynamics.gravity_vector

        # ----------------------------------------------
        # A. PPO chooses an action
        # ----------------------------------------------
        action, log_prob, value = agent.select_action(observation)

        # action contains 6 joint angles, aka q_desired
        q = robot.robot_state.get_joint_angles()
        # print("joint angles:", q)
        q_dot = robot.robot_state.get_joint_velocities()
        # print("Joint velocities:", q_dot)

        position_error = action - q
        velocity_error = q_dot_desired - q_dot

        desired_accel = q_ddot_desired + Kp @ position_error + Kd @ velocity_error
        torques = M @ desired_accel + C + g

        torque_limits = np.array([150, 150, 150, 28, 28, 28])
        torques = np.clip(
            torques,
            -torque_limits,
            torque_limits
        )
        # ----------------------------------------------
        # B. Apply torque to robot
        # ----------------------------------------------
        robot.control_torques(torques)

        # ----------------------------------------------
        # C. Advance physics
        # ----------------------------------------------
        robot.sim.step()

        # ----------------------------------------------
        # D. Read NEW state after physics step
        # ----------------------------------------------
        q = robot.robot_state.get_joint_angles()
        q_dot = robot.robot_state.get_joint_velocities()
        ee_location = robot.robot_state.get_end_effector_position()

        # ----------------------------------------------
        # E. Create next observation
        # ----------------------------------------------
        new_observation = create_observation(
            q,
            q_dot,
            ee_location,
            desired_end_effector_pos
        )

        # ----------------------------------------------
        # F. Calculate reward
        # ----------------------------------------------
        reward = calculate_reward(
            ee_location,
            desired_end_effector_pos
        )

        # ----------------------------------------------
        # G. Check if episode is finished
        # ----------------------------------------------
        distance = np.linalg.norm(
            desired_end_effector_pos - ee_location
        )

        done = distance < 0.05

        # ----------------------------------------------
        # H. Store reward for the action we just took
        # ----------------------------------------------
        buffer_index = agent._step - 1

        agent.store_reward_done(
            buffer_index,
            reward,
            done
        )

        # ----------------------------------------------
        # I. Move to next observation
        # ----------------------------------------------
        observation = new_observation

        # ----------------------------------------------
        # J. PPO update after collecting n_steps
        # ----------------------------------------------
        if agent._step == agent.n_steps:

            info = agent.update(
                observation,
                done
            )

            print(
                f"Step: {step} | "
                f"Policy loss: {info['policy_loss']:.4f} | "
                f"Value loss: {info['value_loss']:.4f} | "
                f"Entropy: {info['entropy']:.4f}"
            )

        # ----------------------------------------------
        # K. Print occasionally, NOT every step
        # ----------------------------------------------
        if step % 1000 == 0:
            print(
                f"Step {step} | "
                f"Distance: {distance:.4f} | "
                f"Reward: {reward:.4f}"
            )
        # if step % 100 == 0:
        #     print("Action:", action)
        #     print("EE:", ee_location)
        #     print("Distance:", distance)



if __name__ == "__main__":
    # with open("torques_with_object.csv", "w") as f:
    #     f.write("")  # Clear the file before starting
    run()