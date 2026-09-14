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

    q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0]) # random desired joint angles for the robot arm
 
    n_joints = len(q_desired)
    Kp = np.diag(np.full(n_joints, 200.0))
    Kd = np.diag(np.full(n_joints, 45.0))

    while True:
        # Access dynamic quantities
        gravity = robot.robot_model.get_dynamics().gravity_vector

        # Access robot state
        q = robot.robot_state.get_joint_angles()
        # print("joint angles:", q)
        # print("Joint angles:", q)
        q_dot = robot.robot_state.get_joint_velocities()
        # print("Joint velocities:", q_dot)

        # PD + gravity compensation control from tutorial slides
        # Could be done simpler but I thought I'd stick with the matrix form to stay consistent with the lectures
        position_error = q_desired - q
        torques = Kp @ position_error - Kd @ q_dot + gravity

        # Send torques
        robot.control_torques(torques)

        # Set gripper finger width
        robot.control_finger_width(0.01)
        robot.sim.step()




if __name__ == "__main__":
    run()
