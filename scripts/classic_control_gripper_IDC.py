import argparse
import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType


def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=False,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot


def run():
    # Initialize simulation and create the scene
    robot = create_scene()

    q_desired = np.array([-1.57, -1.2, 1.8, -1.57, -1.57, 0.0])
    q_dot_desired = np.zeros_like(q_desired)     # static setpoint -> zero desired velocity
    q_ddot_desired = np.zeros_like(q_desired)    # static setpoint -> zero desired acceleration

    n_joints = len(q_desired)

    Kp = np.diag(np.full(n_joints, 200.0))
    Kd = np.diag(np.full(n_joints, 45.0))

    while True:
        # Access dynamic quantities
        dynamics = robot.robot_model.get_dynamics()
        M = dynamics.mass_matrix
        C = dynamics.coriolis_vector
        g = dynamics.gravity_vector

        # Access robot state
        q = robot.robot_state.get_joint_angles()
        q_dot = robot.robot_state.get_joint_velocities()

        # Computed torque / inverse dynamics control:
        #   tau = M(q) [ q_ddot_d + Kp*(q_d - q) + Kd*(q_dot_d - q_dot) ] + C(q, q_dot) + g(q)
        position_error = q_desired - q
        velocity_error = q_dot_desired - q_dot
        desired_accel = q_ddot_desired + Kp @ position_error + Kd @ velocity_error

        torques = M @ desired_accel + C + g

        # Send torques
        robot.control_torques(torques)

        # Set gripper finger width
        robot.control_finger_width(0.01)
        robot.sim.step()


if __name__ == "__main__":
    run()