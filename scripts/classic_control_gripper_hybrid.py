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
from scipy.spatial.transform import Rotation as R, Slerp

J_prev = None
dt = 1.0 / 240.0

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

def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sdot = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    sddot = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, sdot, sddot

def create_waypoint(pos, quat, duration, gripper_width, Kp, Kd):
    return {
        "pos": np.asarray(pos, dtype=float),
        "quat": np.asarray(quat, dtype=float),
        "duration": duration,
        "gripper_width": gripper_width,
        "Kp": np.asarray(Kp, dtype=float),
        "Kd": np.asarray(Kd, dtype=float),
    }


def pose_error(pos, quat, pos_d, quat_d):
    e_pos = pos_d - pos
    R_err = R.from_quat(quat_d) * R.from_quat(quat).inv()
    e_ori = R_err.as_rotvec()
    return np.concatenate([e_pos, e_ori])


def step(robot, pos_d, quat_d, vel_d, ang_vel_d, acc_d, ang_acc_d, Kp, Kd, width):
    global J_prev, dt

    dynamics = robot.robot_model.get_dynamics()
    M = dynamics.mass_matrix
    C = dynamics.coriolis_vector
    g = dynamics.gravity_vector

    q_dot = robot.robot_state.get_joint_velocities()
    J = robot.robot_model.get_jacobian()

    pos_ee = robot.robot_state.get_end_effector_position()
    rot_ee = robot.robot_state.get_end_effector_orientation()
    vel_ee = J @ q_dot

    e = pose_error(pos_ee, rot_ee, pos_d, quat_d)
    vel_d_full = np.concatenate([vel_d, ang_vel_d])
    acc_d_full = np.concatenate([acc_d, ang_acc_d])
    e_dot = vel_d_full - vel_ee

    # Cartesian inertia at the end effector
    Lam = np.linalg.pinv(J @ np.linalg.solve(M, J.T))

    if J_prev is not None:
        Jdot_qdot = ((J - J_prev) / dt) @ q_dot
    else:
        Jdot_qdot = np.zeros(6)

    a_cmd = acc_d_full + Kd * e_dot + Kp * e - Jdot_qdot
    F = Lam @ a_cmd
    J_prev = J.copy()

    torques = J.T @ F + C + g

    # Write torques in csv file
    with open("torques.csv", "a") as f:
        f.write(",".join(map(str, torques)) + "\n")

    robot.control_torques(torques)
    robot.control_finger_width(width)
    robot.sim.step()

    return e

def interpolate(wp_start, wp_end, t, T):
    tau = t / T
    s, sdot, sddot = min_jerk(tau)

    dpos = wp_end['pos'] - wp_start['pos']
    pos_d = wp_start['pos'] + s * dpos
    vel_d = sdot / T * dpos
    acc_d = sddot / T**2 * dpos

    r0 = R.from_quat(wp_start['quat'])
    r1 = R.from_quat(wp_end['quat'])
    slerp = Slerp([0, 1], R.concatenate([r0, r1]))
    quat_d = slerp(s).as_quat()

    rotvec = (r1 * r0.inv()).as_rotvec()
    ang_vel_d = sdot / T * rotvec
    ang_acc_d = sddot / T**2 * rotvec

    Kp = wp_start['Kp'] + s * (wp_end['Kp'] - wp_start['Kp'])
    Kd = wp_start['Kd'] + s * (wp_end['Kd'] - wp_start['Kd'])
    width = wp_start['gripper_width'] + s * (wp_end['gripper_width'] - wp_start['gripper_width'])

    return pos_d, quat_d, vel_d, ang_vel_d, acc_d, ang_acc_d, Kp, Kd, width


def run():
    # Initialize simulation and create the scene
    robot, cube, tray = create_scene()
    robot.control_finger_width(0.1)
    i = 0
    dt = 1.0 / 240.0

    # Gains. With INERTIA_SHAPING these act on a unit mass (acceleration level),
    # so D = 2*sqrt(K) gives critical damping.
    HIGH_K = np.array([800., 800., 800., 60., 60., 60.])
    HIGH_D = 2 * np.sqrt(HIGH_K)
    GRIP = 0.5
    loop = True
    grasp_performed = False
    drop_performed = False
    location_reached = False
    imp_location_reached = False
    while True:
        q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0])  # random desired joint angles for the robot arm
        # Desired location for end effector
        desired_end_effector_pos = [ [0.7, 0.5, 0.3], [0.7, 0.5, 0.16]]
        desired_end_effector_orientation = [0, 0.7071, 0, 0.7071]
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




        # Approach object closely with inverse kinematics
        while not location_reached:
            # Access dynamic quantities
            dynamics = robot.robot_model.get_dynamics()
            M = dynamics.mass_matrix
            C = dynamics.coriolis_vector
            g = dynamics.gravity_vector
            end_effector_position = robot.robot_state.get_end_effector_position()
            J = robot.robot_model.get_jacobian()
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

            # Send torques
            robot.control_torques(torques)

            # check if location of end effector is reached
            cosine = (np.dot(end_effector_position, desired_end_effector_pos[i]) /
                      (norm(end_effector_position) * norm(desired_end_effector_pos[i])))
            #print(end_effector_position, desired_end_effector_pos)
            print(cosine)
            end_effector_velocity = robot.robot_state.get_end_effector_linear_velocity()

            if i == 1 and cosine >= 0.998:
                location_reached = True
            if cosine >= 0.99999 and all(v < 0.01 for v in end_effector_velocity):
                location_reached = True
            robot.sim.step()

        print("asdiasda")

        pos0 = robot.robot_state.get_end_effector_position()
        quat0 = robot.robot_state.get_end_effector_orientation()
        quat_down = np.array(desired_end_effector_orientation)
        waypoints = [
            create_waypoint(pos0, quat0, 0.0, GRIP, HIGH_K, HIGH_D),
            create_waypoint([0.7, 0.5, 0.16], quat_down, 7.0, 0.01, HIGH_K, HIGH_D),
            create_waypoint([-0.7, 0.5, 0.5], quat_down, 7.0, 0.01, HIGH_K, HIGH_D)
        ]
        while not imp_location_reached:

            # Run each segment once
            for wp_a, wp_b in zip(waypoints[:-1], waypoints[1:]):
                print(wp_a, wp_b)
                print("moving to", wp_b['pos'])
                T = wp_b['duration']
                n_steps = int(round(T / dt))
                for i in range(n_steps + 1):
                    t = i * dt
                    step(robot, *interpolate(wp_a, wp_b, t, T))
            imp_location_reached = True


        while not grasp_performed:
            robot.control_finger_width(0.1)
            # This causes the arm to hit the tray, but it keeps the sim up so i left it in
            robot.sim.step()

if __name__ == "__main__":
    run()