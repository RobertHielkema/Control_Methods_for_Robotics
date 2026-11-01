"""
Pick-and-place environment for the UR7e (PyBullet).

Same recipe as the button-press env that worked: the policy outputs joint deltas,
the inverse dynamics controller tracks them, and the task is split into phases
with one waypoint each.

  phase 0  approach : go above the cube            (gripper open)
  phase 1  descend  : go down to the grasp pose    (gripper open)
                      -> reaching it triggers a SCRIPTED gripper close
  phase 2  lift     : go back above the cube       (gripper closed, cube in hand)
  phase 3  carry    : go above the tray            (gripper closed)
                      -> reaching it triggers a SCRIPTED release, then the episode ends

The gripper open/close is scripted on purpose: learning *when* to close from scratch
is a hard exploration problem and the phases already tell the arm where to do it.
"""
import os
import numpy as np
import pybullet as p
from ur_simulation.classic_control.robots.ur7e import UR7e

# --------------------------------------------------
# Control setup (same as the button env that worked)
# --------------------------------------------------
ACTION_REPEAT = 20       # physics steps per policy step (1 kHz sim -> 50 Hz policy)
ACTION_SCALE = 0.1       # max joint change (rad) per policy step
EPISODE_STEPS = 1500     # policy steps per episode (scripted gripper holds are not counted)
KP = 50.0
KD = 20.0
DYNAMICS_EVERY = 1       # recompute M, C, g every N physics steps. 4 is ~4x faster, slightly less exact.
START_NOISE = 0.0        # rad, random perturbation of the start pose. Raise to ~0.05 once it works.

# --------------------------------------------------
# Scene geometry (taken from classic_control_gripper_with_object.py)
# --------------------------------------------------
Q_NEUTRAL = np.array([1.57, -1.7, 2.4, -1.57, -1.57, -1.57])
TOOL_DOWN_QUAT = np.array([0.0, 0.7071, 0.0, 0.7071])   # (x, y, z, w)

CUBE_HALF = 0.01
CUBE_MASS = 0.2
CUBE_FRICTION = 1.0      # PyBullet default is 0.5, which makes a 2 cm cube slip out of the fingers
CUBE_START = np.array([0.7, 0.5, CUBE_HALF + 0.001])    # resting on the floor
TRAY_POS = [-0.7, 0.5, 0.0]

ABOVE_CUBE = np.array([0.7, 0.5, 0.3])
GRASP = np.array([0.7, 0.5, 0.16])
TRAY_ABOVE = np.array([-0.7, 0.5, 0.5])

FINGER_OPEN = 0.1        # open_grasp() in your script
FINGER_CLOSED = 0.01     # close_grasp() in your script
GRIPPER_HOLD_STEPS = 25  # policy steps (0.5 s) the arm holds still while the gripper closes
RELEASE_HOLD_STEPS = 40  # policy steps (0.8 s) the arm holds still while the cube falls and settles

# --------------------------------------------------
# Phases
# --------------------------------------------------
PHASE_NAMES = ["approach", "descend", "lift", "carry"]
PHASE_TARGETS = [ABOVE_CUBE, GRASP, ABOVE_CUBE, TRAY_ABOVE]
PHASE_TOL = [0.08, 0.02, 0.08, 0.08]     # m, distance at which the phase is completed
ORI_WEIGHTS = [6.0, 6.0, 6.0, 1.0]       # carry is low: the tray may be too far for a vertical tool
N_PHASES = len(PHASE_NAMES)

# --------------------------------------------------
# Reward parameters (starting guesses, tune these)
# --------------------------------------------------
PROGRESS_WEIGHT = 8.0
PHASE_BONUS = 400.0      # one-off bonus per completed phase
PLACE_BONUS = 1000.0     # cube ended up inside the tray
DROP_PENALTY = 300.0     # cube slipped out of the gripper, or was knocked away before the grasp
HOLD_BONUS = 3.0         # per policy step while the cube is held and lifted
LIFT_HEIGHT = 0.03       # m above the floor that counts as lifted
DROP_TOL = 0.04          # m, allowed change of the cube-to-gripper offset after the grasp
KNOCK_TOL = 0.05         # m, cube displacement before the grasp that counts as knocked away

OBS_DIM = 29             # 6 q + 6 q_dot + 3 ee + 3 target-ee + 4 phase + 3 orientation err + 3 cube-ee + 1 gripper


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
    """Shortest rotation from q_current to q_desired in the world frame: (rotation_vector, angle)."""
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


class PickPlaceEnv:
    def __init__(self, start_phase: int = 0, realtime: bool = False):
        """
        start_phase=0: start at the neutral pose (full task).
        start_phase=1: start above the cube (descend, grasp, lift, carry, release).
        start_phase=2: start with the cube already grasped (lift, carry, release). The grasp is
                       done by a scripted routine at every reset, so a reset takes a bit longer.
        realtime=True: sleep every physics step (for watching/demos).
        """
        assert start_phase in (0, 1, 2)
        self.start_phase = start_phase
        self.realtime = realtime

        self.robot = UR7e(block_gripper=False, neutral_joints=Q_NEUTRAL)
        self.robot_id = self.robot.robot_id
        self.ee_index = self.robot.robot_state.ee_index

        p.loadURDF("plane.urdf")
        self.tray_id = p.loadURDF("tray/traybox.urdf", basePosition=TRAY_POS, useFixedBase=True)
        self.cube_id = self._create_cube()

        # Tray volume, used to decide whether the cube ended up inside
        aabb = p.getAABB(self.tray_id)
        self.tray_min = np.array(aabb[0])
        self.tray_max = np.array(aabb[1])
        print(f"Tray AABB min {np.round(self.tray_min, 3)} max {np.round(self.tray_max, 3)} "
              f"(cube counts as placed when it is inside this box)")

        q_min, q_max = [], []
        for idx in self.robot.arm_indices:
            info = p.getJointInfo(self.robot_id, int(idx))
            q_min.append(info[8])
            q_max.append(info[9])
        self.q_min = np.array(q_min)
        self.q_max = np.array(q_max)

        self.Kp = np.diag(np.full(6, KP))
        self.Kd = np.diag(np.full(6, KD))

        # Joint configurations for the waypoints (IK, refined iteratively). Errors are printed.
        self.q_above, self.ref_quat = self._solve_ik("above cube", ABOVE_CUBE)
        self.q_grasp, _ = self._solve_ik("grasp pose", GRASP)
        self._solve_ik("above tray", TRAY_ABOVE)   # only reported

        self.phase = 0
        self.t = 0
        self.prev_dist = 0.0
        self.grasped = False
        self.gripper_closed = False
        self.grasp_offset = np.zeros(3)
        self.cube_start = CUBE_START.copy()
        self._set_gripper(False)

    # --------------------------------------------------
    # Scene helpers
    # --------------------------------------------------
    def _create_cube(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[CUBE_HALF] * 3, rgbaColor=[0.9, 0.2, 0.2, 1])
        cube = p.createMultiBody(
            baseMass=CUBE_MASS,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=CUBE_START.tolist(),
        )
        p.changeDynamics(cube, -1, lateralFriction=CUBE_FRICTION, spinningFriction=0.01)
        return cube

    def _reset_cube(self):
        # Your scripted scene spawns the cube at z=0.1 and lets it fall. Here it starts on the floor.
        p.resetBasePositionAndOrientation(self.cube_id, CUBE_START.tolist(), [0, 0, 0, 1])
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0])
        self.cube_start = CUBE_START.copy()

    def _cube_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.cube_id)[0])

    def _cube_in_tray(self):
        c = self._cube_pos()
        m = 0.02
        inside_xy = (self.tray_min[0] + m < c[0] < self.tray_max[0] - m and
                     self.tray_min[1] + m < c[1] < self.tray_max[1] - m)
        return bool(inside_xy and c[2] < self.tray_max[2])

    def _set_gripper(self, closed: bool):
        self.gripper_closed = closed
        self.robot.control_finger_width(FINGER_CLOSED if closed else FINGER_OPEN)

    # --------------------------------------------------
    # Robot helpers
    # --------------------------------------------------
    @property
    def target(self):
        return PHASE_TARGETS[self.phase]

    def _ee_pose(self):
        # computeForwardKinematics=True so the pose is valid right after a reset
        s = p.getLinkState(self.robot_id, self.ee_index, computeForwardKinematics=True)
        return np.array(s[4]), np.array(s[5])

    def _read_state(self):
        q = self.robot.robot_state.get_joint_angles()
        q_dot = self.robot.robot_state.get_joint_velocities()
        ee, quat = self._ee_pose()
        return q, q_dot, ee, quat

    def _solve_ik(self, name, position, quat=TOOL_DOWN_QUAT, n_iter=100):
        """PyBullet IK starts from the current joint state, so re-solving from the previous
        solution converges to a much more accurate answer than a single call."""
        sol = None
        for _ in range(n_iter):
            sol = np.array(self.robot.robot_model.get_inverse_kinematics(position, quat))[:6]
            self.robot.set_joint_angles(sol)
        ee, achieved_quat = self._ee_pose()
        pos_err = np.linalg.norm(ee - position)
        _, ori_err = orientation_error(achieved_quat, quat)
        print(f"IK {name}: position error {pos_err * 100:.2f} cm, orientation error {ori_err:.3f} rad")
        if pos_err > 0.02 or ori_err > 0.1:
            print(f"  WARNING: {name} is not reachable with the tool pointing down "
                  f"(or IK did not converge). Check the waypoint.")
        return sol, achieved_quat

    def _run_controller(self, q_target, n_substeps):
        """Inverse dynamics controller tracking q_target for n_substeps physics steps."""
        dyn = None
        for i in range(n_substeps):
            if dyn is None or i % DYNAMICS_EVERY == 0:
                dyn = self.robot.robot_model.get_dynamics()
            q = self.robot.robot_state.get_joint_angles()
            q_dot = self.robot.robot_state.get_joint_velocities()

            desired_accel = self.Kp @ (q_target - q) - self.Kd @ q_dot
            torques = dyn.mass_matrix @ desired_accel + dyn.coriolis_vector + dyn.gravity_vector

            self.robot.control_torques(torques)   # clips to the motor limits
            self.robot.sim.step(realtime=self.realtime)

    # --------------------------------------------------
    # Scripted gripper routines
    # --------------------------------------------------
    def _close_gripper(self):
        """Hold the arm still, close the fingers, remember where the cube sits relative to the gripper."""
        q_hold = self.robot.robot_state.get_joint_angles().copy()
        self._set_gripper(True)
        self._run_controller(q_hold, GRIPPER_HOLD_STEPS * ACTION_REPEAT)
        ee, _ = self._ee_pose()
        self.grasp_offset = self._cube_pos() - ee
        self.grasped = True

    def _release(self):
        """Hold the arm still, open the fingers, let the cube fall. Returns True if it ended up in the tray."""
        q_hold = self.robot.robot_state.get_joint_angles().copy()
        self._set_gripper(False)
        self.grasped = False
        self._run_controller(q_hold, RELEASE_HOLD_STEPS * ACTION_REPEAT)
        return self._cube_in_tray()

    # --------------------------------------------------
    # Observation / reward
    # --------------------------------------------------
    def _obs(self, q, q_dot, ee, ori_vec, cube):
        phase_onehot = np.zeros(N_PHASES)
        phase_onehot[self.phase] = 1.0
        return np.concatenate([
            q / np.pi,
            q_dot * 0.1,
            ee,
            self.target - ee,
            phase_onehot,
            ori_vec,
            cube - ee,
            [float(self.gripper_closed)],
        ]).astype(np.float32)

    def _reward(self, dist, prev_dist, ori_angle):
        w_1, t_e = 0.001, 0.0001
        r = -(w_1 * dist ** 2 + np.log(dist ** 2 + t_e))     # log distance term
        r += PROGRESS_WEIGHT * (prev_dist - dist)             # progress toward the current waypoint
        r -= ORI_WEIGHTS[self.phase] * ori_angle              # keep the tool pointing down
        return r

    # --------------------------------------------------
    # Gym-style API
    # --------------------------------------------------
    def reset(self):
        self._reset_cube()
        self.grasped = False
        self.grasp_offset = np.zeros(3)
        self._set_gripper(False)

        noise = np.random.uniform(-START_NOISE, START_NOISE, 6)
        if self.start_phase == 0:
            q0 = Q_NEUTRAL + noise
            self.phase = 0
        elif self.start_phase == 1:
            q0 = self.q_above + noise
            self.phase = 1
        else:
            q0 = self.q_grasp
            self.phase = 1
        self.robot.set_joint_angles(np.clip(q0, self.q_min, self.q_max))
        self.t = 0

        if self.start_phase == 2:
            # let the fingers open and the arm settle, then grasp with the scripted routine
            self._run_controller(self.q_grasp.copy(), 10 * ACTION_REPEAT)
            self._close_gripper()
            self.phase = 2

        q, q_dot, ee, quat = self._read_state()
        cube = self._cube_pos()
        self.prev_dist = float(np.linalg.norm(self.target - ee))
        ori_vec, _ = orientation_error(quat, self.ref_quat)
        return self._obs(q, q_dot, ee, ori_vec, cube)

    def step(self, action):
        q = self.robot.robot_state.get_joint_angles()
        q_target = np.clip(q + ACTION_SCALE * np.clip(action, -1.0, 1.0), self.q_min, self.q_max)
        self._run_controller(q_target, ACTION_REPEAT)

        q, q_dot, ee, quat = self._read_state()
        cube = self._cube_pos()
        dist = float(np.linalg.norm(self.target - ee))
        ori_vec, ori_angle = orientation_error(quat, self.ref_quat)

        reward = self._reward(dist, self.prev_dist, ori_angle)
        done = False
        dropped = False
        knocked = False
        placed = False

        # ---- failure checks ----
        if self.grasped:
            offset_err = float(np.linalg.norm((cube - ee) - self.grasp_offset))
            if offset_err > DROP_TOL:                      # cube slipped out of the gripper
                reward -= DROP_PENALTY
                dropped = True
                done = True
        elif self.phase <= 1:
            if np.linalg.norm(cube[:2] - self.cube_start[:2]) > KNOCK_TOL:   # pushed the cube away
                reward -= DROP_PENALTY
                knocked = True
                done = True

        held = bool(self.grasped and not done and cube[2] > self.cube_start[2] + LIFT_HEIGHT)
        if held:
            reward += HOLD_BONUS

        # ---- phase transitions ----
        if not done and dist < PHASE_TOL[self.phase]:
            reward += PHASE_BONUS
            if self.phase == 1:               # at the grasp pose: close the gripper, then lift
                self._close_gripper()
                self.phase = 2
            elif self.phase == 3:             # above the tray: release, then the episode ends
                placed = self._release()
                if placed:
                    reward += PLACE_BONUS
                done = True
            else:
                self.phase += 1
            # the scripted routines moved the sim forward, refresh everything
            q, q_dot, ee, quat = self._read_state()
            cube = self._cube_pos()
            ori_vec, ori_angle = orientation_error(quat, self.ref_quat)
            dist = float(np.linalg.norm(self.target - ee))   # distance to the NEW target

        self.prev_dist = dist
        self.t += 1
        if self.t >= EPISODE_STEPS:
            done = True                       # time limit

        obs = self._obs(q, q_dot, ee, ori_vec, cube)
        info = {
            "dist": dist,
            "phase": self.phase,
            "held": held,
            "placed": placed,
            "dropped": dropped,
            "knocked": knocked,
            "ori_angle": ori_angle,
            "cube": cube,
            "ee": ee,
        }
        return obs, reward, done, info
