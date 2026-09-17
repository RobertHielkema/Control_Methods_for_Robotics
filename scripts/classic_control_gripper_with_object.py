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


def open_grasp(robot:UR7e):
    # Open fingers
    robot.control_finger_width(0.1)

    # COntrol finger forcers - idk how tho

def close_grasp(robot:UR7e):
    # Close fingers
    robot.control_finger_width(0.01)

def axisangle_to_q(theta, v):
    x = v[0]
    y = v[1]
    z = v[2]

    w = cos(theta/2.)
    x = x * sin(theta/2.)
    y = y * sin(theta/2.)
    z = z * sin(theta/2.)
    return[x, y, z, w]

def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=False,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)

    # Spawn tray or smt idk
    tray_id = p.loadURDF("tray/traybox.urdf", basePosition=[-0.7, 0.5, 0])

    # add a cube to simulation
    cube_size = 0.01

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[cube_size, cube_size, cube_size]
    )

    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[cube_size, cube_size, cube_size]
    )

    cube_id = p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0.7, 0.5, 0.1]
    )

    return robot, cube_id, tray_id


def run():
    # Initialize simulation and create the scene
    robot, cube, tray = create_scene()
    robot.control_finger_width(0.1)
    i = 0
    grasp_performed = False
    drop_performed = False
    while True:
        q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0])  # random desired joint angles for the robot arm
        # Desired location for end effector
        desired_end_effector_pos = [[0.7, 0.5, 0.3], [0.7, 0.5, 0.16], [-0.7, 0.5, 0.5]]
        # desired_end_effector_orientation = axisangle_to_q(90, [0, 1, 0])
        desired_end_effector_orientation = [0, 0.7071, 0, 0.7071]
        print(desired_end_effector_orientation)
        q_dot_desired = np.zeros_like(q_desired)  # static setpoint -> zero desired velocity
        q_ddot_desired = np.zeros_like(q_desired)
        n_joints = 6
        if i == 0:
            Kp = np.diag(np.full(n_joints, 200))
            Kd = np.diag(np.full(n_joints, 45))
        elif i == 1:
            Kp = np.diag(np.full(n_joints, 5))
            Kd = np.diag(np.full(n_joints, 1))
        elif i == 2:
            Kp = np.diag(np.full(n_joints, 5))
            Kd = np.diag(np.full(n_joints, 3))


        # Kp = np.diag([50, 50, 50, 30, 30, 30])
        # Kd = np.diag([14, 14, 14, 10, 10, 10])

        # Compute inverse kinematics to get desired joint angles
        q_desired = robot.robot_model.get_inverse_kinematics(position=desired_end_effector_pos[i],
                                                             quaternion=desired_end_effector_orientation)
        q_desired = np.array(q_desired).tolist()
        q_desired = q_desired[:6]  # Only take the first 6 joint angles for the arm
        # max and min joint limits
        q_desired[0] = np.clip(q_desired[0], -2.5 * np.pi, 2.5 * np.pi)
        q_desired[1] = np.clip(q_desired[1], -2.5 * np.pi, 2.5 * np.pi)
        q_desired[2] = np.clip(q_desired[2], -2.5 * np.pi, 2.5 * np.pi)
        q_desired[3] = np.clip(q_desired[3], -2.5 * np.pi, 2.5 * np.pi)
        q_desired[4] = np.clip(q_desired[4], -2.5 * np.pi, 2.5 * np.pi)
        q_desired[5] = np.clip(q_desired[5], -2.5 * np.pi, 2.5 * np.pi)
        location_reached = False

        while not location_reached:
            # Access dynamic quantities
            dynamics = robot.robot_model.get_dynamics()
            M = dynamics.mass_matrix
            C = dynamics.coriolis_vector
            g = dynamics.gravity_vector
            end_effector_position = robot.robot_state.get_end_effector_position()
            end_effector_orientation = robot.robot_state.get_end_effector_orientation()

            # Access dynamic quantities
            gravity = robot.robot_model.get_dynamics().gravity_vector



            # Access robot state
            q = robot.robot_state.get_joint_angles()
            # print("joint angles:", q)
            q_dot = robot.robot_state.get_joint_velocities()
            # print("Joint velocities:", q_dot)

            position_error = q_desired - q
            velocity_error = q_dot_desired - q_dot

            desired_accel = q_ddot_desired + Kp @ position_error + Kd @ velocity_error
            torques = M @ desired_accel + C + g

            if i == 2:
                cube_gravity = - 0.2 # u can also add * 10 for g
                torques = M @ desired_accel + C + g + cube_gravity

            torque_limits = np.array([150, 150, 150, 28, 28, 28])
            torques = np.clip(
                torques,
                -torque_limits,
                torque_limits
            )

            # Send torques
            robot.control_torques(torques)

            # check if location of end effector is reached
            cosine = (np.dot(end_effector_position, desired_end_effector_pos[i]) /
                      (norm(end_effector_position) * norm(desired_end_effector_pos[i])))
            print(end_effector_position, desired_end_effector_pos)
            print(cosine)
            end_effector_velocity = robot.robot_state.get_end_effector_linear_velocity()

            if i == 2 and cosine >= 0.998:
                location_reached = True
            if cosine >= 0.99999 and all(v < 0.01 for v in end_effector_velocity):
                location_reached = True
            robot.sim.step()
        # # Grasping the object
        if not grasp_performed and i == 1:
            close_grasp(robot)
            grasp_performed = True
            robot.sim.step()
        else:
            open_grasp(robot)

        if not drop_performed and i == 2:
            robot.control_finger_width(0.1)
            drop_performed = True
        # Switch to 2nd location
        if i < 2:
            i += 1

if __name__ == "__main__":
    run()