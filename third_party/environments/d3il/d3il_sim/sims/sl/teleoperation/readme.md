# Teleoperation system

This package implements the teleoperation from one Franka Panda robot to another. Currently, there are 3 different modes
available: 
- `demo_teacher.py` allows the control of the panda robot in zero-force mode. Therefore, this script does not teleoperate another 
  robot. 
- `demo_teleopration.py` is the main script for teleoperation. The primary robot which is controlled by a human sends its
  position and velocity to the replica robot which follows it with a PD-controller. A further feature of the teleoperation is the 
  **force feedback**: when the replica robot receives an external force (e.g. by moving against an obstacle), the primary
  robot simulates the same force in a controlled manner. 
- `demo_virtualtwin.py` is the script to control a virtual robot in Mujoco with a primary real robot. Right now, 
  force feedback is not implemented here.