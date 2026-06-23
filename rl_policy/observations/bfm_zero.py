from .base import Observation

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

class base_ang_vel(Observation):
    def compute(self, scale: float = 1.0) -> np.ndarray:
        # base_ang_vel = self.state_processor.root_ang_vel_b   # for medium-size model and large-size model without dr
        base_ang_vel = self.state_processor.root_ang_vel_b  # large-size models with dr0109
        return base_ang_vel * scale

class projected_gravity(Observation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.v = np.array([0, 0, -1])

    def compute(self, scale: float = 1.0) -> np.ndarray:
        base_quat = self.state_processor.root_quat_b
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        return projected_gravity * scale

class dof_pos_minus_default(Observation):
    def compute(self, scale: float = 1.0) -> np.ndarray:
        return (self.state_processor.joint_pos - self.env.default_dof_angles) * scale

class dof_vel(Observation):
    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.state_processor.joint_vel * scale

class prev_actions(Observation):
    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.env.last_action * scale
    

class reference_trajectory(Observation):
    def __init__(self, env, scale = 1, **kwargs):
        super().__init__(env, scale, **kwargs)

###### HISTORY REDO LATER
class base_ang_vel_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.base_ang_vel_history = np.zeros((self.steps, 3))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.base_ang_vel_history = np.roll(self.base_ang_vel_history, 1, axis=0)
        self.base_ang_vel_history[0, :] = self.state_processor.root_ang_vel_b

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.base_ang_vel_history.reshape(-1) * scale
    
class projected_gravity_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.projected_gravity_history = np.zeros((self.steps, 3))
        self.v = np.array([0, 0, -1])
    
    def update(self, data: Dict[str, Any]) -> None:
        base_quat = self.state_processor.root_quat_b
        projected_gravity = quat_rotate_inverse_numpy(
            base_quat[None, :], 
            self.v[None, :]
        ).squeeze(0)
        self.projected_gravity_history = np.roll(self.projected_gravity_history, 1, axis=0)
        self.projected_gravity_history[0, :] = projected_gravity

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.projected_gravity_history.reshape(-1) * scale

class dof_pos_minus_default_history(Observation):
    def __init__(self, steps: int, default_pos: list, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.default_pos = np.array(default_pos)
        self.dof_pos_minus_default_history = np.zeros((self.steps, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.dof_pos_minus_default_history = np.roll(self.dof_pos_minus_default_history, 1, axis=0)
        self.dof_pos_minus_default_history[0, :] = self.state_processor.joint_pos - self.default_pos

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.dof_pos_minus_default_history.reshape(-1) * scale
    
class dof_vel_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps
        self.dof_vel_history = np.zeros((self.steps, self.state_processor.num_dof))
    
    def update(self, data: Dict[str, Any]) -> None: 
        self.dof_vel_history = np.roll(self.dof_vel_history, 1, axis=0)
        self.dof_vel_history[0, :] = self.state_processor.joint_vel
    
    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.dof_vel_history.reshape(-1) * scale

class prev_actions_history(Observation):
    def __init__(self, steps: int, **kwargs):
        super().__init__(**kwargs)
        self.steps = steps 
        self.prev_actions = np.zeros((self.steps, self.env.num_actions))
    
    def update(self, data: Dict[str, Any]) -> None:
        self.prev_actions = np.roll(self.prev_actions, 1, axis=0)
        self.prev_actions[0, :] = data["action"]

    def compute(self, scale: float = 1.0) -> np.ndarray:
        return self.prev_actions.reshape(-1) * scale
    