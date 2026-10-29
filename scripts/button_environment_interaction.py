# # Programmatically generate the URDF for a Base + Sensor Block
# # The block is attached via a fixed joint at index 0
# urdf_str = """
# <robot name="force_sensor_block">
#   <!-- Main Base Structure -->
#   <link name="base_link">
#     <inertial>
#       <mass value="10.0"/>
#       <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
#     </inertial>
#     <visual>
#       <geometry><box size="0.2 0.2 0.2"/></geometry>
#       <material name="blue"><color rgba="0 0 0.8 1"/></material>
#     </visual>
#     <collision>
#       <geometry><box size="0.2 0.2 0.2"/></geometry>
#     </collision>
#   </link>

#   <!-- The Small Block for Force Measurement -->
#   <link name="sensor_block">
#     <inertial>
#       <mass value="0.01"/> <!-- Keep mass low to reduce inertial noise -->
#       <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
#     </inertial>
#     <visual>
#       <geometry><box size="0.05 0.05 0.02"/></geometry>
#       <material name="red"><color rgba="0.8 0 0 1"/></material>
#     </visual>
#     <collision>
#       <geometry><box size="0.05 0.05 0.02"/></geometry>
#     </collision>
#   </link>

#   <!-- Fixed Joint holding the sensor pad -->
#   <joint name="sensor_joint" type="fixed">
#     <parent link="base_link"/>
#     <child link="sensor_block"/>
#     <origin xyz="0 0 0.11" rpy="0 0 0"/>
#   </joint>
# </robot>
# """

# # Save and load the URDF string dynamically
# with open("sensor_block.urdf", "w") as f:
#     f.write(urdf_str)

import argparse
import time
import numpy as np
import pybullet as p
import pybullet_data
import time
from ur_simulation.classic_control.robots.ur7e import UR7e
from ur_simulation.pybullet import PyBullet
from ur_simulation.classic_control.robot_state.pybullet_robot_state import JointType
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
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # # add a cube to simulation
    # cube_size = 0.1  # half-extent = 10 cm

    # collision_shape = p.createCollisionShape(
    #     p.GEOM_BOX,
    #     halfExtents=[cube_size, cube_size, cube_size]
    # )

    # visual_shape = p.createVisualShape(
    #     p.GEOM_BOX,
    #     halfExtents=[cube_size, cube_size, cube_size]
    # )

    # cube_id = p.createMultiBody(
    #     baseMass=1,
    #     baseCollisionShapeIndex=collision_shape,
    #     baseVisualShapeIndex=visual_shape,
    #     basePosition=[1, 1, 0.1]
    # )

    cube_id = p.loadURDF("sensor_block.urdf", basePosition=[0.5, 0.4, 0.1], useFixedBase=True)
    joint_index = 0  # "sensor_joint"
    p.enableJointForceTorqueSensor(cube_id, joint_index, enableSensor=True)


    return robot, cube_id, joint_index

def run():
    # Initialize simulation and create the scene
    robot, cube_id, joint_index = create_scene()
    robot.control_finger_width(0.1)
    i = 0
    grasp_performed = False
    drop_performed = False
    step = 0
    while True:
        q_desired = np.array([0.0, -1.2, 1.8, -1.57, -1.57, 0.0])  # random desired joint angles for the robot arm
        # Desired location for end effector
        desired_end_effector_pos = [[0.5, 0.4, 0.5], [0.5, 0.4, 0.2], [-0.7, 0.5, 0.5]]
        # desired_end_effector_orientation = axisangle_to_q(90, [0, 1, 0])
        desired_end_effector_orientation = [0, 0.7071, 0, 0.7071]
        #print(desired_end_effector_orientation)
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
            step += 1
            joint_feedback = p.getJointState(cube_id, joint_index)
            reaction_forces = joint_feedback[2] 
            
            # Print telemetry every 50 time steps
            if step % 50 == 0:
                fx, fy, fz, mx, my, mz = reaction_forces
                print(f"Step {step:04d} | Measured Force: [{fx:6.2f}, {fy:6.2f}, {fz:6.2f}] N | Torque: [{mx:6.2f}, {my:6.2f}, {mz:6.2f}] Nm")
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

            # Write torques in csv file
            with open("torques_with_object.csv", "a") as f:
                f.write(",".join(map(str, torques)) + "\n")

            # Send torques
            robot.control_torques(torques)
            # print(f"Da roques: {torques}")
            # check if location of end effector is reached
            cosine = (np.dot(end_effector_position, desired_end_effector_pos[i]) /
                      (norm(end_effector_position) * norm(desired_end_effector_pos[i])))
            #print(end_effector_position, desired_end_effector_pos)
            #print(cosine)
            end_effector_velocity = robot.robot_state.get_end_effector_linear_velocity()

            if i == 2 and cosine >= 0.998:
                location_reached = True
            if cosine >= 0.99999 and all(v < 0.01 for v in end_effector_velocity):
                location_reached = True
            robot.sim.step()
        # # Grasping the object
        if not grasp_performed and i == 1:
            grasp_performed = True
            robot.sim.step()

        if not drop_performed and i == 2:
            drop_performed = True

        if drop_performed and i == 2:
            quit()
        # Switch to 2nd location
        if i < 2:
            i += 1

if __name__ == "__main__":
    with open("torques_with_object.csv", "w") as f:
        f.write("")  # Clear the file before starting
    run()