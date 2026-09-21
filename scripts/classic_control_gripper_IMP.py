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
    return s, sdot

def interpolate(wp_start, wp_end, t, T):
    tau = t / T
    s, sdot = min_jerk(tau)

    pos_d = wp_start['pos'] + s * (wp_end['pos'] - wp_start['pos'])
    vel_d = sdot / T * (wp_end['pos'] - wp_start['pos'])

    key_rots = R.from_quat([wp_start['quat'], wp_end['quat']])
    slerp = Slerp([0, 1], key_rots)
    quat_d = slerp(s).as_quat()

    rotvec = (R.from_quat(wp_end['quat']) * R.from_quat(wp_start['quat']).inv()).as_rotvec()
    ang_vel_d = sdot / T * rotvec

    Kp = wp_start['Kp'] + s * (wp_end['Kp'] - wp_start['Kp'])
    Kd = wp_start['Kd'] + s * (wp_end['Kd'] - wp_start['Kd'])

    return pos_d, quat_d, vel_d, ang_vel_d, Kp, Kd


def create_waypoint(pos, quat, duration, gripper_width, Kp, Kd):
    pos = np.asarray(pos, dtype=float)        
    quat = np.asarray(quat, dtype=float)       
    duration = duration                      
    gripper_width = gripper_width
    Kp = np.asarray(Kp, dtype=float)           
    Kd = np.asarray(Kd, dtype=float)
    return  {"pos": pos, "quat": quat, "duration": duration, "gripper_witdh": gripper_width, "Kp": Kp, "Kd": Kd}

def pose_error(pos, quat, pos_d, quat_d):
    e_pos = pos_d - pos
    R_err = R.from_quat(quat_d) * R.from_quat(quat).inv()
    e_ori = R_err.as_rotvec() 
    return np.concatenate([e_pos, e_ori])


def run():
    loop = False
    # Initialize simulation and create the scene
    robot = create_scene()

    q_desired = np.array([0.5, 0.0, 0.5, -1.57, -1.57, 0.0])
    q_desired2 = np.array([0.5, 0.0, 0.1, -1.57, -1.57, 0.0])
    q_desired3 = np.array([0.5, 0.0, 0.5, -1.57, -1.57, 0.0])
    q_dot_desired = np.zeros_like(q_desired)     # static setpoint -> zero desired velocity
    q_ddot_desired = np.zeros_like(q_desired)    # static setpoint -> zero desired acceleration

    n_joints = len(q_desired)

    Kp = np.diag(np.full(n_joints, 200.0))
    Kd = np.diag(np.full(n_joints, 45.0))

    HIGH_K = np.array([800., 800., 800., 60., 60., 60.])   
    HIGH_D = 2 * np.sqrt(HIGH_K)
    
    SOFT_K = np.array([300., 300., 150., 30., 30., 30.])   
    SOFT_D = 2 * np.sqrt(SOFT_K)

    dt = 1.0 / 240.0  
    quat_down  = R.from_euler("xyz", [np.pi, 0, 0]).as_quat()
    
    waypoints = []
    waypoint = create_waypoint(q_desired[:3], quat_down, 7.0, 0.04, HIGH_K, HIGH_D)
    waypoint2 = create_waypoint(q_desired2[:3], quat_down, 5.0, 0.04, HIGH_K, HIGH_D)
    waypoint3 = create_waypoint(q_desired3[:3], quat_down, 5.0, 0.04, HIGH_K, HIGH_D)
    waypoints.append(waypoint)
    waypoints.append(waypoint2)
    waypoints.append(waypoint3)
    while True:

        for seg, index in enumerate(range(len(waypoints) -1)):
            if index == 1 and not loop:
                loop = True
            elif loop:
                break
            wp_a, wp_b = waypoints[seg], waypoints[seg + 1]
            print(wp_b['pos'])
            T = wp_b['duration']
            t = 0.0

            while t < T:
                pos_d, quat_d, vel_d, ang_vel_d, Kp, Kd = interpolate(wp_a, wp_b, t, T)

                # Access dynamic quantities
                dynamics = robot.robot_model.get_dynamics()
                M = dynamics.mass_matrix
                C = dynamics.coriolis_vector
                g = dynamics.gravity_vector

                # Access robot state
                q = robot.robot_state.get_joint_angles()
                q_dot = robot.robot_state.get_joint_velocities()
                J = robot.robot_model.get_jacobian()

                pos_ee = robot.robot_state.get_end_effector_position()
                rot_ee = robot.robot_state.get_end_effector_orientation()
                vel_ee = J @ q_dot

                e = pose_error(pos_ee, rot_ee, pos_d, quat_d)
                # print(e)
                vel_d_full = np.concatenate([vel_d, ang_vel_d])
                e_dot = vel_d_full - vel_ee
                # print(e_dot)

                F = Kp * e + Kd * e_dot

                torques = J.T @ F + C + g
                

                # Computed torque / inverse dynamics control:
                #   tau = M(q) [ q_ddot_d + Kp*(q_d - q) + Kd*(q_dot_d - q_dot) ] + C(q, q_dot) + g(q)
                # position_error = q_desired - q
                # velocity_error = q_dot_desired - q_dot
                # desired_accel = q_ddot_desired + Kp @ position_error + Kd @ velocity_error

                # torques = M @ desired_accel + C + g

                # Send torques
                robot.control_torques(torques)

                # Set gripper finger width
                robot.control_finger_width(0.01)
                robot.sim.step()
                t += dt


if __name__ == "__main__":
    run()