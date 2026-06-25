import numpy as np
from typing import Any, Dict
from utils.math import quat_rotate_inverse_numpy, matrix_from_quat

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

def motion_anchor_ori_b(env):
    base_quat = env.robot.state.root_quat_b
    ori = quat_rotate_inverse_numpy(
        base_quat[None, :], 
        env.command_manager.anchor_quat_w[None, :]
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)[0]

def generated_commands(env, name):
    return env.command_manager.commands(name)