# Control Methods for Robotics: Model-Based and Model-Free Manipulation

An experimental comparison of **model-based** and **model-free** control paradigms on a 6-DoF Universal Robots **UR7e** arm simulated in **PyBullet**. 

Developed for the *Control Methods for Robotics* course at the **University of Groningen** (Department of Artificial Intelligence).

**Authors:**
* Robert Hielkema
* Vasile Vlad-Mihnea
* Daan de Groot

## Project Overview

This repository benchmarks classical model-based controllers, reinforcement learning and physics-informed neural network controllers across three primary robotic manipulation tasks:

1. **Free Movement (Reaching):** Moving the end-effector to randomly sampled target points within the reachable workspace.
2. **Pick-and-Place:** Approaching, grasping a PyBullet cube object, and transporting it into a drop-off tray.
3. **Force Application (Button Pressing):** Reaching a custom URDF force-sensing button mounted on a block and applying sustained, controlled force.

### Implemented Methods
* **Model-Based**:
  * PD Control with Gravity Compensation
  * Inverse Dynamics (Computed Torque) Control
  * Cartesian Impedance Control
  * Hybrid Inverse Dynamics / Impedance Control
  * Time-Optimal Trajectory Tracking
* **Model-Free / Learned**:
  * PPO (Proximal Policy Optimization) with Gaussian policy and Simulated Annealing parameter tuning
  * Lagrangian Neural Networks (LNN) trained via supervised learning on Inverse Dynamics trajectory data

### Our Report
