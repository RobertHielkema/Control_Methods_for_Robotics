import classic_control_gripper_IMP_fixed as imp
from scipy.spatial.transform import Rotation as R
import numpy as np
import pybullet as p
import time

HIGH_K = np.array([800., 800., 800., 60., 60., 60.])
HIGH_D = 2 * np.sqrt(HIGH_K)
QUAT_DOWN = np.array([0, 0.7071, 0, 0.7071])

def impedance_imitation_generator(robot, target_pos, target_quat=QUAT_DOWN,
                        T=4.0, settle=1.0, render=False):
    """Simulate the impedance controller, return joint trajectory"""
    if not render:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    state_id = p.saveState()
    imp.J_prev = None

    pos0 = robot.robot_state.get_end_effector_position()
    quat0 = robot.robot_state.get_end_effector_orientation()
    wp_a = imp.create_waypoint(pos0, quat0, 0.0, 0.1, HIGH_K, HIGH_D)
    wp_b = imp.create_waypoint(target_pos, target_quat, T, 0.1, HIGH_K, HIGH_D)

    q_ref = []
    n = int(round((T + settle) / imp.dt))
    for i in range(n):
        t = min(i * imp.dt, T)
        imp.step(robot, *imp.interpolate(wp_a, wp_b, t, T))
        if render:
            time.sleep(imp.dt)  
        q_ref.append(np.array(robot.robot_state.get_joint_angles()[:6]))

    final_err = np.linalg.norm(robot.robot_state.get_end_effector_position() - target_pos)

    p.restoreState(state_id)
    p.removeState(state_id)
    imp.J_prev = None
    if not render:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    return np.array(q_ref), final_err