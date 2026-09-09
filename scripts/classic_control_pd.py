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
        block_gripper=True,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot

def run():
    robot = create_scene()
    kp = np.array([80, 80, 80, 35, 35, 35])
    kd = np.array([30, 30, 30, .1, .1, .1])

    waypoints = [
        np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57]),
        np.array([0, -1.57, 1.57, -1.57, -1.57, -1.57]),
    ]
    current_waypoint = 0
    finger_width = 0
    while True:
        waypoint = waypoints[current_waypoint]
        gravity = robot.robot_model.get_dynamics().gravity_vector
        angles = robot.robot_state.get_joint_angles(JointType.REVOLUTE)
        velocities = robot.robot_state.get_joint_velocities(JointType.REVOLUTE)

        pos_error = waypoint - angles
        vel_error =  - velocities

        torques = kp * pos_error + kd * vel_error + gravity

        robot.control_torques(torques)
        robot.sim.step()

        if np.linalg.norm(pos_error) < 0.1:
            current_waypoint = (current_waypoint + 1) % len(waypoints)
            # robot.control_finger_width(finger_width)
            finger_width = .05 - finger_width




if __name__ == "__main__":
    run()
