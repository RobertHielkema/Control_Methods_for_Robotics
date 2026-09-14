import argparse
import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType
from XboxController import XboxController


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

    q_desired = np.array([-1.57, -1.2, 1.8, -1.57, -1.57, 0.0]) # random desired joint angles for the robot arm
 
    n_joints = len(q_desired)
    finger_width = 0.01  # Desired finger width in meters

    # tuned for normal control
    # Kp = np.diag(np.full(n_joints, 200.0))
    # Kd = np.diag(np.full(n_joints, 45.0))

    # tuned for controller
    Kp = np.diag(np.full(n_joints, 1000.0))
    Kd = np.diag(np.full(n_joints, 90.0))

    controller = XboxController()

    while True:
        # Read controller input
        x1, y1, x2, y2, z1, z2, bl, br = controller.read()
        # print(f"Controller input: x={x}, y={y}")
        # print(f"Controller input: z1={z1}, z2={z2}")
        if x1 > 0.1 or x1 < -0.1:
            q_desired[0] = q_desired[0] + x1 * 0.005
        if y1 > 0.1 or y1 < -0.1:
            q_desired[1] = q_desired[1] + y1 * 0.005
        if x2 > 0.1 or x2 < -0.1:
            q_desired[4] = q_desired[4] + x2 * 0.005
        if y2 > 0.1 or y2 < -0.1:
            q_desired[3] = q_desired[3] + y2 * 0.005
        if z1 > 0.1 or z1 < -0.1:
            q_desired[5] = q_desired[5] - z1 * 0.002
        if z2 > 0.1 or z2 < -0.1:
            q_desired[5] = q_desired[5] + z2 * 0.002
        if bl > 0.1:
            finger_width = finger_width - 0.002
        if br > 0.1:
            finger_width = finger_width + 0.002

        # max and min joint limits
        q_desired[0] = np.clip(q_desired[0], -2.5*np.pi, 2.5*np.pi)
        q_desired[1] = np.clip(q_desired[1], -2.5*np.pi, 2.5*np.pi)
        q_desired[2] = np.clip(q_desired[2], -2.5*np.pi, 2.5*np.pi)
        q_desired[3] = np.clip(q_desired[3], -2.5*np.pi, 2.5*np.pi)
        q_desired[4] = np.clip(q_desired[4], -2.5*np.pi, 2.5*np.pi)
        q_desired[5] = np.clip(q_desired[5], -2.5*np.pi, 2.5*np.pi)

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
        robot.control_finger_width(finger_width)
        robot.sim.step()




if __name__ == "__main__":
    run()
