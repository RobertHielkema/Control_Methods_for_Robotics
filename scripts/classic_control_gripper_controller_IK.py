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

    q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0]) # random desired joint angles for the robot arm
 
    n_joints = len(q_desired)
    Kp = np.diag(np.full(n_joints, 200.0))
    Kd = np.diag(np.full(n_joints, 40.0))

    controller = XboxController()

    while True:
        # Read controller input
        x1, y1, x2, y2, z1, z2, _, _ = controller.read()
        end_effector_position = robot.robot_state.get_end_effector_position()
        end_effector_orientation = robot.robot_state.get_end_effector_orientation()

        # Update end-effector position based on controller input
        end_effector_position[0] += -x1 * 0.1  # Move along x-axis
        end_effector_position[1] += y1 * 0.1  # Move along y-axis
        end_effector_position[2] += z1 * 0.1  # Move along z-axis
        end_effector_position[2] += -z2 * 0.1  # Move along z-axis

        # Update end-effector orientation in quaternion based on controller input
        end_effector_orientation[0] += -x2 * 0.1  # Rotate around x-axis
        end_effector_orientation[1] += y2 * 0.1  # Rotate around y-axis

        # Compute inverse kinematics to get desired joint angles
        q_desired = robot.robot_model.get_inverse_kinematics(position=end_effector_position, quaternion=end_effector_orientation)
        q_desired = np.array(q_desired).tolist()
        q_desired = q_desired[:6]  # Only take the first 6 joint angles for the arm
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
