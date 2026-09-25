import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R, Slerp

from time_optimal_trajectory import (
    compute_profile,
    compute_profile_synced,
    evaluate,
    MultiJointTrajectory,
)

# Limits for the generator -- segment durations come out of these instead
# of being hardcoded like the min_jerk version.
# TODO: swap in the UR7e's real limits
N_JOINTS = 6
A_MAX = np.full(N_JOINTS, 3.0)      # rad/s^2
V_MAX = np.full(N_JOINTS, 1.5)      # rad/s
TORQUE_LIMITS = np.array([150, 150, 150, 28, 28, 28])

LIN_A_MAX = 0.6    # m/s^2
LIN_V_MAX = 0.3    # m/s
ANG_A_MAX = 3.0    # rad/s^2
ANG_V_MAX = 1.5    # rad/s
DWELL_TIME = 0.5   # for segments where nothing actually moves (e.g. just closing the gripper)

HIGH_K = np.array([800., 800., 800., 60., 60., 60.])
HIGH_D = 2 * np.sqrt(HIGH_K)   # critically damped

GRIP_OPEN = 0.1
GRIP_CLOSED = 0.01

# this is the orientation that actually points the gripper down -- NOT
# the same thing as R.from_euler("xyz", [pi, 0, 0]), learned that one
# the hard way
QUAT_DOWN = np.array([0, 0.7071, 0, 0.7071])

DT = 1.0 / 240.0
J_prev = None


def create_scene():
    init_joint_angles = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
    robot = UR7e(
        block_gripper=False,
        neutral_joints=init_joint_angles,
    )
    robot.sim.create_plane(0)

    tray_id = p.loadURDF("tray/traybox.urdf", basePosition=[-0.7, 0.5, 0])

    # small cube to pick up
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


def create_waypoint(pos, quat, gripper_width, Kp, Kd):
    return {
        "pos": np.asarray(pos, dtype=float),
        "quat": np.asarray(quat, dtype=float),
        "gripper_width": gripper_width,
        "Kp": np.asarray(Kp, dtype=float),
        "Kd": np.asarray(Kd, dtype=float),
    }


class TimeOptimalCartesianSegment:
    """Drives one waypoint-to-waypoint move. Position and orientation
    each get their own scalar time-optimal profile (0 -> 1), synced so
    they land together, and everything else (gripper width, gains)
    rides along on whichever of those is actually moving."""

    def __init__(self, wp_start, wp_end, aM_lin, vM_lin, aM_ang, vM_ang, eps=1e-6):
        self.wp_start = wp_start
        self.wp_end = wp_end

        self.dpos = wp_end["pos"] - wp_start["pos"]
        self.dist_lin = float(norm(self.dpos))

        r0 = R.from_quat(wp_start["quat"])
        r1 = R.from_quat(wp_end["quat"])
        self.rotvec = (r1 * r0.inv()).as_rotvec()
        self.dist_ang = float(norm(self.rotvec))
        self.slerp = Slerp([0, 1], R.concatenate([r0, r1])) if self.dist_ang > eps else None

        # how long each path takes on its own
        tf_lin = tf_ang = 0.0
        if self.dist_lin > eps:
            tf_lin = compute_profile(0.0, 1.0, 0.0, 0.0,
                                      aM_lin / self.dist_lin, vM_lin / self.dist_lin)["tf"]
        if self.dist_ang > eps:
            tf_ang = compute_profile(0.0, 1.0, 0.0, 0.0,
                                      aM_ang / self.dist_ang, vM_ang / self.dist_ang)["tf"]

        no_motion = self.dist_lin <= eps and self.dist_ang <= eps
        self.tf = max(tf_lin, tf_ang, DWELL_TIME if no_motion else 1e-6)

        # then stretch the faster one so both finish at the same time
        self.prof_lin = None
        if self.dist_lin > eps:
            self.prof_lin = compute_profile_synced(
                0.0, 1.0, aM_lin / self.dist_lin, vM_lin / self.dist_lin, self.tf
            )
        self.prof_ang = None
        if self.dist_ang > eps:
            self.prof_ang = compute_profile_synced(
                0.0, 1.0, aM_ang / self.dist_ang, vM_ang / self.dist_ang, self.tf
            )

    def evaluate(self, t):
        t = min(max(t, 0.0), self.tf)

        if self.prof_lin is not None:
            s, sdot, _ = evaluate(self.prof_lin, t)
        else:
            s, sdot = 1.0, 0.0
        pos_d = self.wp_start["pos"] + s * self.dpos
        vel_d = sdot * self.dpos

        if self.prof_ang is not None:
            sa, sadot, _ = evaluate(self.prof_ang, t)
            quat_d = self.slerp(float(np.clip(sa, 0.0, 1.0))).as_quat()
        else:
            sa, sadot = 1.0, 0.0
            quat_d = self.wp_end["quat"]
        ang_vel_d = sadot * self.rotvec

        # not feeding acceleration in as feedforward -- bang-bang accel
        # jumps around at every switch, and that's what was making it
        # snap. gains are stiff enough to track pos/vel fine without it
        acc_d = np.zeros(3)
        ang_acc_d = np.zeros(3)

        # gripper width / gains should track how far along the move we
        # are, not just elapsed time -- the bang-bang curve isn't linear
        # in time so those two drift apart otherwise (this is what made
        # the gripper close early)
        if self.prof_lin is not None:
            progress = s
        elif self.prof_ang is not None:
            progress = sa
        else:
            progress = t / self.tf if self.tf > 1e-9 else 1.0

        Kp = self.wp_start["Kp"] + progress * (self.wp_end["Kp"] - self.wp_start["Kp"])
        Kd = self.wp_start["Kd"] + progress * (self.wp_end["Kd"] - self.wp_start["Kd"])
        width = self.wp_start["gripper_width"] + progress * (
            self.wp_end["gripper_width"] - self.wp_start["gripper_width"]
        )
        return pos_d, quat_d, vel_d, ang_vel_d, acc_d, ang_acc_d, Kp, Kd, width

    def done(self, t, tol=1e-3):
        return t >= self.tf - tol


def pose_error(pos, quat, pos_d, quat_d):
    e_pos = pos_d - pos
    R_err = R.from_quat(quat_d) * R.from_quat(quat).inv()
    e_ori = R_err.as_rotvec()
    return np.concatenate([e_pos, e_ori])


def step(robot, pos_d, quat_d, vel_d, ang_vel_d, acc_d, ang_acc_d, Kp, Kd, width):
    global J_prev

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
        Jdot_qdot = ((J - J_prev) / DT) @ q_dot
    else:
        Jdot_qdot = np.zeros(6)

    a_cmd = acc_d_full + Kd * e_dot + Kp * e - Jdot_qdot
    F = Lam @ a_cmd
    J_prev = J.copy()

    torques = J.T @ F + C + g
    robot.control_torques(torques)
    robot.control_finger_width(width)
    robot.sim.step()

    return e


def run_joint_space_approach(robot, target_pos, target_quat):
    q_target = robot.robot_model.get_inverse_kinematics(
        position=target_pos, quaternion=target_quat
    )
    q_target = np.clip(np.array(q_target[:6]), -2.5 * np.pi, 2.5 * np.pi)

    q_now = np.array(robot.robot_state.get_joint_angles())
    qd_now = np.array(robot.robot_state.get_joint_velocities())

    traj = MultiJointTrajectory(q_now, q_target, qd_now, np.zeros(N_JOINTS), aM=A_MAX, vM=V_MAX)
    print(f"joint-space approach: {traj.tf:.2f}s")

    Kp = np.diag(np.full(N_JOINTS, 200))
    Kd = np.diag(np.full(N_JOINTS, 45))

    t_start = time.time()
    while True:
        t = time.time() - t_start
        # no qddot feedforward, same reason as the Cartesian segment
        q_des, qd_des, _ = traj.evaluate(t)

        dynamics = robot.robot_model.get_dynamics()
        M = dynamics.mass_matrix
        C = dynamics.coriolis_vector
        g = dynamics.gravity_vector
        q = robot.robot_state.get_joint_angles()
        q_dot = robot.robot_state.get_joint_velocities()

        pos_err = q_des - q
        vel_err = qd_des - q_dot
        desired_accel = Kp @ pos_err + Kd @ vel_err
        torques = M @ desired_accel + C + g
        torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)

        robot.control_torques(torques)
        robot.sim.step()

        if traj.done(t):
            break


def run_cartesian_pick_and_place(robot):
    pos0 = robot.robot_state.get_end_effector_position()
    quat0 = robot.robot_state.get_end_effector_orientation()

    waypoints = [
        create_waypoint(pos0, quat0, GRIP_OPEN, HIGH_K, HIGH_D),                       # start
        create_waypoint([0.7, 0.5, 0.30], QUAT_DOWN, GRIP_OPEN, HIGH_K, HIGH_D),       # pre-grasp
        create_waypoint([0.7, 0.5, 0.16], QUAT_DOWN, GRIP_OPEN, HIGH_K, HIGH_D),       # descend, gripper still open
        create_waypoint([0.7, 0.5, 0.16], QUAT_DOWN, GRIP_CLOSED, HIGH_K, HIGH_D),     # close, arm not moving
        create_waypoint([0.7, 0.5, 0.30], QUAT_DOWN, GRIP_CLOSED, HIGH_K, HIGH_D),     # lift
        create_waypoint([-0.7, 0.5, 0.50], QUAT_DOWN, GRIP_CLOSED, HIGH_K, HIGH_D),    # carry over to the tray
        create_waypoint([-0.7, 0.5, 0.50], QUAT_DOWN, GRIP_OPEN, HIGH_K, HIGH_D),      # drop it
    ]

    for wp_a, wp_b in zip(waypoints[:-1], waypoints[1:]):
        seg = TimeOptimalCartesianSegment(wp_a, wp_b, LIN_A_MAX, LIN_V_MAX, ANG_A_MAX, ANG_V_MAX)
        print("moving to", wp_b["pos"], f"({seg.tf:.2f}s)")
        n_steps = int(round(seg.tf / DT)) + 1
        for k in range(n_steps):
            t = k * DT
            step(robot, *seg.evaluate(t))


def run():
    robot, cube, tray = create_scene()
    robot.control_finger_width(GRIP_OPEN)

    run_joint_space_approach(robot, target_pos=[0.7, 0.5, 0.30], target_quat=QUAT_DOWN)
    run_cartesian_pick_and_place(robot)

    quit()


if __name__ == "__main__":
    run()
