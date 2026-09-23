import argparse
import time
import numpy as np
import pybullet as p
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from scipy.spatial.transform import Rotation as R, Slerp
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType

J_prev = None
dt = 1.0 / 240.0

def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=False,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)
    return robot

def min_jerk(tau):
    tau = np.clip(tau, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sdot = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    sddot = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, sdot, sddot


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


def run():
    robot = create_scene()
    dt = 1.0 / 240.0

    # Gains. With INERTIA_SHAPING these act on a unit mass (acceleration level),
    # so D = 2*sqrt(K) gives critical damping.
    HIGH_K = np.array([800., 800., 800., 60., 60., 60.])
    HIGH_D = 2 * np.sqrt(HIGH_K)

    quat_down = R.from_euler("xyz", [np.pi, 0, 0]).as_quat()
    GRIP = 0.01

    # Start the trajectory from where the robot actually is
    pos0 = robot.robot_state.get_end_effector_position()
    quat0 = robot.robot_state.get_end_effector_orientation()

    waypoints = [
        create_waypoint(pos0, quat0, 0.0, GRIP, HIGH_K, HIGH_D),
        create_waypoint([0.5, 0.0, 0.5], quat_down, 7.0, GRIP, HIGH_K, HIGH_D),
        create_waypoint([0.5, 0.0, 0.1], quat_down, 5.0, GRIP, HIGH_K, HIGH_D),
        create_waypoint([0.5, 0.0, 0.5], quat_down, 5.0, GRIP, HIGH_K, HIGH_D),
    ]

    # Run each segment once
    for wp_a, wp_b in zip(waypoints[:-1], waypoints[1:]):
        print("moving to", wp_b['pos'])
        T = wp_b['duration']
        n_steps = int(round(T / dt))
        for i in range(n_steps + 1):
            t = i * dt
            step(robot, *interpolate(wp_a, wp_b, t, T))

    last = waypoints[-1]
    zeros3 = np.zeros(3)
    quit()


if __name__ == "__main__":
    run()
