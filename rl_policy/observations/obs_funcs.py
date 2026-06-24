import numpy as np
from typing import Any, Dict
from utils.math import quat_rotate_inverse_numpy

def base_ang_vel(env):
    return env.robot.state.root_ang_vel_b

def joint_pos_rel(env):
    return env.robot.state.joint_pos - env.robot.state.default_dof_angles

def joint_vel(env):
    return env.robot.state.joint_vel

def projected_gravity(env):
    base_quat = env.robot.state.root_quat_b
    v = np.array([0, 0, -1])
    return quat_rotate_inverse_numpy(
        base_quat[None, :], 
        v[None, :]
    ).squeeze(0)

def last_action(env):
    return env.last_action