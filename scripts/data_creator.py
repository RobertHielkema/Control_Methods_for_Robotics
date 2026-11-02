
import sys
import os
import time
from pathlib import Path
import numpy as np
import pybullet as p

from ur_simulation.classic_control.robots.ur7e import UR7e

DT = 0.001 # got this from pybullet
NUM_SEGMENTS = 300
# some guys on github used this
MIN_SEGMENT_DURATION = 2.0
MAX_SEGMENT_DURATION = 4.0
# robot's reachable workspace
WORKSPACE_LOW = np.array([-0.85, -0.85, 0])
WORKSPACE_HIGH = np.array([0.85, 0.85, 0.85])
# Keeping the end effector pointing down to make training easier
EE_ORIENTATION = np.array([0.0, 0.7071, 0.0, 0.7071])

# IK solutions that are too far from the current configuration resulted in super high val loss
MAX_IK_JOINT_JUMP = 2.0
KP = np.diag([100.0, 100.0, 100.0, 40.0, 30.0, 20.0])
KD = np.diag([25.0, 25.0, 25.0, 12.0, 10.0, 8.0])
TORQUE_LIMITS = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
OUTPUT_PATH = Path("ur7e_dynamics_dataset.npz")
RANDOM_SEED = 42

def create_scene():
    initial_q = np.array(
        [1.57, -1.7, 2.4, -1.57, -1.57, -1.57],
        dtype=float,
    )

    robot = UR7e(
        block_gripper=False,
        neutral_joints=initial_q,
    )

    robot.sim.create_plane(0)
    tray_path = Path("tray/traybox.urdf")
    if tray_path.exists():
        p.loadURDF(
            str(tray_path),
            basePosition=[-0.7, 0.5, 0.0],
        )

    cube_size = 0.01

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[cube_size] * 3,
    )
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[cube_size] * 3,
    )

    p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[0.7, 0.5, 0.1],
    )

    robot.control_finger_width(0.1)
    return robot
def get_arm_state(robot):
    q = np.asarray(
        robot.robot_state.get_joint_angles(),
        dtype=np.float64,
    ).reshape(-1)[:6]

    q_dot = np.asarray(
        robot.robot_state.get_joint_velocities(),
        dtype=np.float64,
    ).reshape(-1)[:6]

    if q.shape != (6,) or q_dot.shape != (6,):
        raise ValueError(
            f"Expected six arm joints, got q={q.shape}, "
            f"q_dot={q_dot.shape}"
        )

    return q.copy(), q_dot.copy()


def wrap_angle_difference(q_target, q_current):
    error = q_target - q_current
    return np.arctan2(np.sin(error), np.cos(error))


def solve_ik(robot, position, q_current):
    try:
        solution = robot.robot_model.get_inverse_kinematics(
            position=position.tolist(),
            quaternion=EE_ORIENTATION.tolist(),
        )
    except Exception:
        return None

    q_ik = np.asarray(solution, dtype=np.float64).reshape(-1)[:6]

    if q_ik.shape != (6,) or not np.all(np.isfinite(q_ik)):
        return None
    q_ik = q_current + wrap_angle_difference(q_ik, q_current)

    if np.any(np.abs(q_ik - q_current) > MAX_IK_JOINT_JUMP):
        return None

    return q_ik
def sample_ik_target(robot, q_current, rng, max_attempts=100):
    for _ in range(max_attempts):
        position = rng.uniform(WORKSPACE_LOW, WORKSPACE_HIGH)

        q_target = solve_ik(robot, position, q_current)

        if q_target is not None:
            return position, q_target

def quintic_trajectory(q_start, q_goal, duration, t):
    s = np.clip(t / duration, 0.0, 1.0)

    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    blend_dot = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
    blend_ddot = (
        60 * s - 180 * s**2 + 120 * s**3
    ) / (duration**2)

    delta_q = q_goal - q_start

    q_desired = q_start + blend * delta_q
    q_dot_desired = blend_dot * delta_q
    q_ddot_desired = blend_ddot * delta_q

    return q_desired, q_dot_desired, q_ddot_desired


def compute_torques(robot, q, q_dot, q_des, q_dot_des, q_ddot_des):
    M, C, g = robot.robot_model.get_dynamics()

    position_error = q_des - q
    velocity_error = q_dot_des - q_dot

    desired_acceleration = (
        q_ddot_des
        + KP @ position_error
        + KD @ velocity_error
    )

    tau_unclipped = M @ desired_acceleration + C + g
    tau_applied = np.clip(
        tau_unclipped,
        -TORQUE_LIMITS,
        TORQUE_LIMITS,
    )

    return tau_applied

def collect_segment(robot, q_goal, segment_id, rng, data):

    q_start, _ = get_arm_state(robot)

    duration = rng.uniform(
        MIN_SEGMENT_DURATION,
        MAX_SEGMENT_DURATION,
    )
    num_steps = int(round(duration / DT))

    for step_idx in range(num_steps):
        t = step_idx * DT

        q_des, q_dot_des, q_ddot_des = quintic_trajectory(
            q_start,
            q_goal,
            duration,
            t,
        )
        q, q_dot = get_arm_state(robot)

        tau = compute_torques(robot, q, q_dot, q_des, q_dot_des, q_ddot_des)
        robot.control_torques(tau)
        robot.sim.step()
        next_q, next_q_dot = get_arm_state(robot)

        data["q"].append(q)
        data["q_dot"].append(q_dot)
        data["action"].append(tau)
        data["next_q"].append(next_q)
        data["next_q_dot"].append(next_q_dot)

        data["q_desired"].append(q_des)
        data["q_dot_desired"].append(q_dot_des)
        data["segment_id"].append(segment_id)
        data["time_in_segment"].append(t)

def save_dataset(data, output_path):
    array_data = {
        key: np.asarray(value)
        for key, value in data.items()
    }

    array_data["dt"] = np.asarray(DT)
    array_data["control_mode"] = np.asarray("torque")
    array_data["seed"] = np.asarray(RANDOM_SEED)

    np.savez_compressed(output_path, **array_data)

    print(f"\nSaved dataset to: {output_path.resolve()}")
    print(f"Number of transitions: {len(data['q'])}")

def run():
    rng = np.random.default_rng(RANDOM_SEED)
    robot = create_scene()

    print("PyBullet physics parameters:")
    print(p.getPhysicsEngineParameters())

    data = {
        "q": [],
        "q_dot": [],
        "action": [],
        "next_q": [],
        "next_q_dot": [],
        "q_desired": [],
        "q_dot_desired": [],
        "segment_id": [],
        "time_in_segment": [],
    }

    try:
        q_current, _ = get_arm_state(robot)

        print("Starting UR7e dynamics data collection.")
        print(f"Initial joint configuration: {q_current}")

        for segment_id in range(NUM_SEGMENTS):
            q_current, _ = get_arm_state(robot)
            target_position, q_goal = sample_ik_target(
                robot,
                q_current,
                rng,
            )

            print(
                f"Segment {segment_id + 1}/{NUM_SEGMENTS}: "
                f"position={np.round(target_position, 3)}, "
                f"joint target={np.round(q_goal, 3)}"
            )

            collect_segment(
                robot=robot,
                q_goal=q_goal,
                segment_id=segment_id,
                rng=rng,
                data=data,
            )
            if (segment_id + 1) % 10 == 0:
                save_dataset(data, OUTPUT_PATH)
                print(
                    f"Completed {segment_id + 1} segments "
                    f"({len(data['q'])} transitions)."
                )

        save_dataset(data, OUTPUT_PATH)

    finally:
        if data["q"]:
            save_dataset(data, OUTPUT_PATH)

        print("Data collection finished.")


if __name__ == "__main__":
    run()
