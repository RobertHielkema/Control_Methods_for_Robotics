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
import torch
from pathlib import Path
from lnn import UR7eLNN

DT = 0.001
DEVICE = torch.device("cpu")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints_lnn" / "best_lnn.pt"


CONTROL_SUBSTEPS = 4

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

    return robot

def lnn_predict_acceleration(model, q, q_dot, tau, dt=DT):
    q_t = torch.as_tensor(
        q, dtype=torch.float32, device=DEVICE
    ).reshape(1, 6)

    qd_t = torch.as_tensor(
        q_dot, dtype=torch.float32, device=DEVICE
    ).reshape(1, 6)

    tau_t = torch.as_tensor(
        tau, dtype=torch.float32, device=DEVICE
    ).reshape(1, 6)

    with torch.enable_grad():
        _, _, qdd_t = model(q_t, qd_t, tau_t, dt)

    return qdd_t.detach().cpu().numpy().reshape(6)


def lnn_inverse_dynamics(model, q, q_dot, desired_accel):
    q_t = torch.as_tensor(
        q, dtype=torch.float32, device=DEVICE
    ).reshape(1, 6).detach().requires_grad_(True)

    qd_t = torch.as_tensor(
        q_dot, dtype=torch.float32, device=DEVICE
    ).reshape(1, 6)

    qdd_t = torch.as_tensor(
        desired_accel, dtype=torch.float32, device=DEVICE
    ).reshape(1, 6)

    with torch.enable_grad():
        M = model.mass_matrix(q_t)
        V = model.potential(q_t)

        gradV = torch.autograd.grad(
            V.sum(), q_t, retain_graph=True
        )[0]

        # dM_dq[:, i, j, k] = dM[j,k]/dq[i]
        dM_dq = torch.zeros(
            (1, 6, 6, 6), dtype=q_t.dtype, device=q_t.device
        )

        for j in range(6):
            for k in range(j, 6):
                grad_entry = torch.autograd.grad(
                    M[:, j, k].sum(),
                    q_t,
                    retain_graph=True,
                )[0]

                for i in range(6):
                    dM_dq[:, i, j, k] = grad_entry[:, i]
                    dM_dq[:, i, k, j] = grad_entry[:, i]

        h = 0.5 * torch.einsum(
            "bj,bijl,bl->bi", qd_t, dM_dq, qd_t
        )

        Dqd = model.damping().unsqueeze(0) * qd_t

        tau = (torch.bmm(M, qdd_t.unsqueeze(-1)).squeeze(-1) + Dqd- h + gradV)

    return tau.detach().cpu().numpy().reshape(6)

def load_lnn():
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = UR7eLNN(
        q_mean=checkpoint["q_mean"],
        q_std=checkpoint["q_std"],
        hidden=128,
        epsilon=1e-3,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded LNN from: {CHECKPOINT_PATH}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.8f}")

    return model



def run():
    model = load_lnn()

    robot = create_scene()
    robot.control_finger_width(0.1)

    i = 0
    grasp_performed = False
    drop_performed = False
    while True:
        q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0])  # random desired joint angles for the robot arm
        # Desired location for end effector
        desired_end_effector_pos = [[0.717, 0.44,  0.281]] #[0.7, 0.4, 0.1] [0.6, 0.4, 0.5] [0.717, 0.44,  0.281]
        # desired_end_effector_orientation = axisangle_to_q(90, [0, 1, 0])
        desired_end_effector_orientation = [0, 0.7071, 0, 0.7071]
        #print(desired_end_effector_orientation)
        q_dot_desired = np.zeros_like(q_desired)  # static setpoint -> zero desired velocity
        q_ddot_desired = np.zeros_like(q_desired)
        n_joints = 6
        Kp = np.diag([100.0, 100.0, 100.0, 40.0, 30.0, 20.0])
        Kd = np.diag([25.0, 25.0, 25.0, 12.0, 10.0, 8.0])


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
            # Read the current robot state
            q = np.asarray(
                robot.robot_state.get_joint_angles(),
                dtype=np.float32,
            )[:6]

            q_dot = np.asarray(
                robot.robot_state.get_joint_velocities(),
                dtype=np.float32,
            )[:6]

            # Desired joint-space acceleration from the PD controller
            position_error = np.asarray(q_desired) - q
            velocity_error = q_dot_desired - q_dot

            desired_accel = q_ddot_desired + Kp @ position_error + Kd @ velocity_error

            # Use the trained LNN to calculate the required torque
            torques = lnn_inverse_dynamics(
                model, q, q_dot, desired_accel
            )

            torques = np.clip(
                torques,
                -np.array([150, 150, 150, 28, 28, 28]),
                np.array([150, 150, 150, 28, 28, 28]),
            )

            robot.control_torques(torques)


            robot.sim.step()

            # Check actual Cartesian distance, not the angle between position vectors
            end_effector_position = np.asarray(
                robot.robot_state.get_end_effector_position()
            )
            end_effector_velocity = np.asarray(
                robot.robot_state.get_end_effector_linear_velocity()
            )

            target_position = np.asarray(desired_end_effector_pos[i])

            position_error_ee = np.linalg.norm(
                end_effector_position - target_position
            )

            speed_ee = np.linalg.norm(end_effector_velocity)

            location_reached = (
                    position_error_ee < 0.005
            )


            print(
                f"Waypoint {i} | "
                f"EE error={position_error_ee:.4f} m | "
                f"EE speed={speed_ee:.4f} m/s | "
                f"torque={np.round(torques, 2)}"
            )

if __name__ == "__main__":
    # with open("torques_with_object.csv", "w") as f:
    #     f.write("")  # Clear the file before starting
    run()
