"""
Button-press environment for the UR7e (PyBullet).

Task: go to the waypoint above the button (phase 0), then push down onto the
button (phase 1). The policy outputs joint deltas, an inverse dynamics
controller tracks them (same setup as the free-motion script that worked).

Reward = your log distance term + progress term + press-force term
         - orientation penalty (+ a bonus when phase 0 is completed)
"""
import os
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e

# --------------------------------------------------
# Control setup (same as the free-motion run that worked)
# --------------------------------------------------
ACTION_REPEAT = 20       # physics steps per policy step (1 kHz sim -> 50 Hz policy)
ACTION_SCALE = 0.1       # max joint change (rad) per policy step
EPISODE_STEPS = 1200
KP = 50.0
KD = 20.0

# --------------------------------------------------
# Scene / task geometry (taken from button_environment_interaction.py)
# --------------------------------------------------
Q_NEUTRAL = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
BUTTON_BASE_POS = [0.5, 0.4, 0.1]
WAYPOINT_ABOVE = np.array([0.5, 0.4, 0.5])   # point above the button
WAYPOINT_PRESS = np.array([0.5, 0.4, 0.2])   # point "in" the button
TOOL_DOWN_QUAT = np.array([0.0, 0.7071, 0.0, 0.7071])  # (x, y, z, w), from your button script
FINGER_WIDTH = 0.01      # closed fingers press like a single probe. Use 0.1 for the old open gripper.

# --------------------------------------------------
# Reward parameters (tune these)
# --------------------------------------------------
REACH_TOL = 0.10         # m, phase 0 -> phase 1 switch distance to WAYPOINT_ABOVE
PROGRESS_WEIGHT = 10.0
PHASE_BONUS = 20.0       # one-off bonus when phase 0 is completed
F_TARGET = 10.0          # N, press force at which the force reward saturates (also "press success")
F_MAX = 40.0             # N, forces above this are penalised (no slamming the button)
FORCE_WEIGHT = 10.0      # max force reward per policy step
ORI_WEIGHT = 3.0         # penalty per radian of tool orientation error

OBS_DIM = 23             # 6 q + 6 q_dot + 3 ee + 3 target-ee + 1 phase + 3 orientation err + 1 force

BUTTON_URDF = """<robot name="force_sensor_block">
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <visual>
      <geometry><box size="0.2 0.2 0.2"/></geometry>
      <material name="blue"><color rgba="0 0 0.8 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.2 0.2 0.2"/></geometry>
    </collision>
  </link>
  <link name="sensor_block">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <geometry><box size="0.05 0.05 0.02"/></geometry>
      <material name="red"><color rgba="0.8 0 0 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.05 0.05 0.02"/></geometry>
    </collision>
  </link>
  <joint name="sensor_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_block"/>
    <origin xyz="0 0 0.11" rpy="0 0 0"/>
  </joint>
</robot>
"""


def ensure_button_urdf(path: str = "sensor_block.urdf") -> str:
    """Write the button URDF next to the script if it doesn't exist yet."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(BUTTON_URDF)
    return path


# --------------------------------------------------
# Quaternion helpers, all quaternions are (x, y, z, w)
# --------------------------------------------------
def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def orientation_error(q_current, q_desired):
    """
    Rotation that takes q_current to q_desired, expressed in the world frame.
    Returns (rotation_vector, angle). The angle is in [0, pi] (shortest rotation).
    """
    q_c = np.asarray(q_current, dtype=float)
    q_d = np.asarray(q_desired, dtype=float)
    q_c = q_c / np.linalg.norm(q_c)
    q_d = q_d / np.linalg.norm(q_d)
    q_err = quat_mul(q_d, quat_conj(q_c))
    if q_err[3] < 0:
        q_err = -q_err
    v = q_err[:3]
    s = np.linalg.norm(v)
    angle = 2.0 * np.arctan2(s, q_err[3])
    vec = np.zeros(3) if s < 1e-9 else angle * v / s
    return vec, float(angle)


class ButtonPressEnv:
    def __init__(self, start_at_waypoint_1: bool = True, realtime: bool = False):
        """
        start_at_waypoint_1=True : episodes start above the button (phase 1, press only).
                                   Much easier, train this first.
        start_at_waypoint_1=False: episodes start near the neutral pose, the arm has to
                                   reach the waypoint above the button first (phase 0).
        realtime=True            : sleep every physics step (for watching/demos).
        """
        self.start_at_waypoint_1 = start_at_waypoint_1
        self.realtime = realtime

        self.robot = UR7e(block_gripper=False, neutral_joints=Q_NEUTRAL)
        self.robot_id = self.robot.robot_id
        self.ee_index = self.robot.robot_state.ee_index
        self.robot.control_finger_width(FINGER_WIDTH)

        p.loadURDF("plane.urdf")
        self.button_id = p.loadURDF(
            ensure_button_urdf(), basePosition=BUTTON_BASE_POS, useFixedBase=True
        )
        # joint 0 = fixed joint between the base box and the sensor pad
        p.enableJointForceTorqueSensor(self.button_id, 0, enableSensor=True)

        q_min, q_max = [], []
        for idx in self.robot.arm_indices:
            info = p.getJointInfo(self.robot_id, int(idx))
            q_min.append(info[8])
            q_max.append(info[9])
        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)

        self.Kp = np.diag(np.full(6, KP))
        self.Kd = np.diag(np.full(6, KD))

        # Joint configuration above the button (IK, refined iteratively)
        self.q_above, self.ref_quat = self._solve_ik(WAYPOINT_ABOVE, TOOL_DOWN_QUAT)

        self.phase = 0
        self.t = 0
        self.prev_dist = 0.0

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @property
    def target(self):
        return WAYPOINT_ABOVE if self.phase == 0 else WAYPOINT_PRESS

    def _ee_pose(self):
        # computeForwardKinematics=True so the pose is valid right after a reset
        s = p.getLinkState(self.robot_id, self.ee_index, computeForwardKinematics=True)
        return np.array(s[4]), np.array(s[5])

    def _read_state(self):
        q = self.robot.robot_state.get_joint_angles()
        q_dot = self.robot.robot_state.get_joint_velocities()
        ee, quat = self._ee_pose()
        return q, q_dot, ee, quat

    def _contact_force(self):
        """Sum of contact normal forces (N) between the robot and the button pad (link 0)."""
        pts = p.getContactPoints(bodyA=self.robot_id, bodyB=self.button_id, linkIndexB=0)
        return float(sum(c[9] for c in pts))

    def _sensor_force(self):
        """Raw reading of the force/torque sensor (fx, fy, fz, mx, my, mz). Sign convention
        depends on the joint frame, so the reward uses the contact normal force instead."""
        return p.getJointState(self.button_id, 0)[2]

    def _solve_ik(self, position, quat, n_iter=100):
        """PyBullet IK starts from the current joint state, so re-solving from the previous
        solution converges to a much more accurate answer than a single call."""
        sol = None
        for _ in range(n_iter):
            sol = np.array(
                self.robot.robot_model.get_inverse_kinematics(position, quat)
            )[:6]
            self.robot.set_joint_angles(sol)
        ee, achieved_quat = self._ee_pose()
        pos_err = np.linalg.norm(ee - position)
        _, ori_err = orientation_error(achieved_quat, quat)
        print(f"IK above button: position error {pos_err * 100:.2f} cm, orientation error {ori_err:.3f} rad")
        if pos_err > 0.02 or ori_err > 0.1:
            print("WARNING: IK did not converge well. Check WAYPOINT_ABOVE / TOOL_DOWN_QUAT.")
        # Use the orientation that was actually achieved as the reference, so the
        # start pose has zero orientation error by construction.
        return sol, achieved_quat

    def _obs(self, q, q_dot, ee, ori_vec, force):
        return np.concatenate([
            q / np.pi,
            q_dot * 0.1,
            ee,
            self.target - ee,
            [float(self.phase)],
            ori_vec,
            [min(force / F_TARGET, 3.0)],
        ]).astype(np.float32)

    def _reward(self, dist, prev_dist, force, ori_angle):
        w_1, t_e = 0.001, 0.0001
        r = -(w_1 * dist ** 2 + np.log(dist ** 2 + t_e))     # your original term
        r += PROGRESS_WEIGHT * (prev_dist - dist)             # progress toward the current target
        r -= ORI_WEIGHT * ori_angle                           # keep the tool pointing down
        if self.phase == 1:
            # reward pressing, saturating at F_TARGET, penalise slamming above F_MAX
            r += FORCE_WEIGHT * min(force, F_TARGET) / F_TARGET
            r -= FORCE_WEIGHT * min(max(force - F_MAX, 0.0) / F_MAX, 1.0)
        return r

    # --------------------------------------------------
    # Gym-style API
    # --------------------------------------------------
    def reset(self):
        if self.start_at_waypoint_1:
            q0 = self.q_above + np.random.uniform(-0.02, 0.02, 6)
            self.phase = 1
        else:
            # q0 = Q_NEUTRAL + np.random.uniform(-0.1, 0.1, 6)
            q0 = Q_NEUTRAL
            self.phase = 0
        self.robot.set_joint_angles(np.clip(q0, self.q_min, self.q_max))

        self.t = 0
        q, q_dot, ee, quat = self._read_state()
        self.prev_dist = float(np.linalg.norm(self.target - ee))
        ori_vec, _ = orientation_error(quat, self.ref_quat)
        return self._obs(q, q_dot, ee, ori_vec, 0.0)

    def step(self, action):
        q = self.robot.robot_state.get_joint_angles()
        q_target = np.clip(
            q + ACTION_SCALE * np.clip(action, -1.0, 1.0), self.q_min, self.q_max
        )

        forces = []
        for _ in range(ACTION_REPEAT):
            dyn = self.robot.robot_model.get_dynamics()
            q = self.robot.robot_state.get_joint_angles()
            q_dot = self.robot.robot_state.get_joint_velocities()

            desired_accel = self.Kp @ (q_target - q) - self.Kd @ q_dot
            torques = dyn.mass_matrix @ desired_accel + dyn.coriolis_vector + dyn.gravity_vector

            self.robot.control_torques(torques)   # clips to the motor limits
            self.robot.sim.step(realtime=self.realtime)
            forces.append(self._contact_force())

        force = float(np.mean(forces))            # mean over the substeps, contact forces are spiky

        q, q_dot, ee, quat = self._read_state()
        dist = float(np.linalg.norm(self.target - ee))
        ori_vec, ori_angle = orientation_error(quat, self.ref_quat)

        reward = self._reward(dist, self.prev_dist, force, ori_angle)

        # Phase switch: reached the point above the button -> go press it
        if self.phase == 0 and dist < REACH_TOL:
            self.phase = 1
            reward += PHASE_BONUS
            dist = float(np.linalg.norm(self.target - ee))   # distance to the NEW target
        self.prev_dist = dist

        self.t += 1
        done = self.t >= EPISODE_STEPS           # time limit only

        obs = self._obs(q, q_dot, ee, ori_vec, force)
        info = {
            "dist": dist,
            "force": force,
            "peak_force": float(max(forces)),
            "sensor_fz": float(self._sensor_force()[2]),
            "phase": self.phase,
            "ori_angle": ori_angle,
            "press_success": bool(self.phase == 1 and force >= F_TARGET),
            "ee": ee,
        }
        return obs, reward, done, info
