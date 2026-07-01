
import os
import onnxruntime
from rl_policy.observations import Observation, ObsGroup
from rl_policy.observations import ObsManager
import numpy as np
from g1_deploy.utils import load_yaml

class ActionManager:
    def __init__(self, env, control_cfg):
        self.act_scale = env.to_tensor(
            control_cfg['action_scale'], env.joint_order)
        self.torque_limits = env.to_tensor(
            control_cfg['torque_limits'], env.joint_order)
        self.stiffness = env.to_tensor(
            control_cfg['stiffness'], env.joint_order)
        self.damping = env.to_tensor(
            control_cfg['damping'], env.joint_order)
        self.torque_limits = env.to_tensor(
            control_cfg['torque_limits'], env.joint_order)
        self.pos_lim_l = env.to_tensor(
            control_cfg['pos_limit_low'], env.joint_order)
        self.pos_lim_h = env.to_tensor(
            control_cfg['pos_limit_high'], env.joint_order)

    def pos_clip(self, pos_val: np.ndarray):
        return np.clip(pos_val, self.pos_lim_l, self.pos_lim_h)
    
    def scale_and_add(self, raw_act, default_joint_pos):
        return raw_act*self.act_scale + default_joint_pos

def interpolate(s, e, num):
    delta = (e-s)/num
    return np.array([
        s + delta*i for i in range(num)
    ])

class WrapperBase:
    def __init__(self, policy_dir_path, robot, device=None):
        self.robot = robot
        self.device = device
        config_path = os.path.join(policy_dir_path, 'config.yaml')
        policy_path = os.path.join(policy_dir_path, 'policy.onnx')
        self.parse_config(config_path)
        if os.path.exists(policy_path):
            self.setup_policy(policy_path)

    def parse_config(self):
        pass
    
    def setup_policy(self, policy_path):
        # load onnx policy
        self.onnx_policy_session = onnxruntime.InferenceSession(policy_path)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
        def policy_act(obs):
            return self.onnx_policy_session.run(
                [self.onnx_output_name], {self.onnx_input_name: obs})[0]
        self.policy = policy_act    

class Zero(WrapperBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.command_set = False
        self.idx=0

    def resample_command(self):
        self.commands = interpolate(
            self.robot.state['joint_pos'], 
            self.standing_pose, 3*50)
        self.command_set = True

    def __call__(self):
        super().__call__()
        if not self.command_set:
            self.resample_command()
        q =self.commands[self.idx]
        self.idx = min(self.idx+1, len(self.commands)-1)
        placeholder = np.zeros_like(q)
        return {
            'q': q,
            'dq': placeholder,
            'tau': placeholder,
            'kp': self.stiffness,
            'kd': self.damping
        }
    
    def reset(self):
        self.command_set = False
        self.idx = 0

class PolicyWrapper(WrapperBase):

    def parse_config(self, config_path):
        self.cfg = load_yaml_conf(config_path)
        self.joint_order = self.cfg["joint_order"]
        self.num_dof = len(self.joint_order)
        self.default_dof_angles = self.to_tensor(
            self.cfg["init_state"]["default_joint_pos"], self.joint_order)
        self.last_action  = np.zeros(len(self.joint_order))
        self.action_manager = ActionManager(self, self.cfg['control'])
        self.obs_manager = ObsManager(self, self.cfg['observations'])
        self.command_manager = CommandManager(self, self.cfg['command']) # change to CommandManager later
        self.setup_observations()
        self.reset_callbacks = []
        self.update_callbacks = []
        for manager in [self.obs_manager, self.command_manager]:
            self.reset_callbacks.append(manager.reset)
            self.update_callbacks.append(manager.update)

    def to_tensor(self, cfg_dict, order, dtype = np.float32):
        return np.asarray(
            [cfg_dict[name] for name in order],
            dtype=dtype)


    def reset(self):
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        for update_callback in self.update_callbacks:
            update_callback()

    def __call__(self):
        obs_d = self.obs_manager.compute()
        raw_action = self.policy(obs_d['policy'])
        self.last_action = raw_action[:]
        action = self.process_action(raw_action)
        return action
    
    def process_action(self, raw_act):
        q =self.action_manager.scale_and_add(raw_act, self.default_dof_angles)
        placeholder = np.zeros_like(q)
        return {
            'q': q,
            'dq': placeholder,
            'tau': placeholder,
            'kp': self.stiffness,
            'kd': self.damping
        }
    
    def set_next_trajectory(self):
        pass

class MotionTopic:
    def __init__(self, topic_name):
        pass


class CommandManager:

    cmnds_map = {
        "base_velocity": VelocityCommand,
        'motion': MotionCommand
    }

    def __init__(self, env, commands_cfg):
        self.env = env
        self.commmands = {
            name:self.cmnds_map[name](cmd_conf)
            for name, cmd_conf in commands_cfg.items()
        }

    def update(self):
        for cmd in self.commmands.values():
            cmd.update()
    
    def reset(self):
        for cmd in self.commmands.values():
            cmd.reset()

class VelocityCommand:
    def __init__(self, env, command_cfg):
        self.controller = env.robot.wireless_controller
        self.ranges = self.command_cfg['ranges']
        self.vel_command_b = np.zeros(3)
    
    def update(self):
        self.vel_command_b[:] = [self.controller.Ly, self.controller.Lx, self.controller.Rx]
        for i, name in enumerate(['lin_vel_x', 'lin_vel_y', 'ang_vel_z']):
            neg, pos = self.ranges[name]
            self.vel_command_b[i] *= neg if self.vel_command_b[i]<0 else pos
    
    def reset(self):
        pass


class MotionCommand:

    def __init__(self, command_cfg):
        self.trajectories = {}
        self.cfg =  command_cfg
        self.cur_trj = None
        self.idx = 0
        self.ended = False
        self.body_ids = command_cfg['body_indexes']
        self.anchor_id = command_cfg['anchor_index']
        
        for traj_path in command_cfg['trajectories']:
            name = os.path.basename(traj_path)
            self.trajectories[name] = self.load_trajectory(traj_path)
        self.names = list(self.trajectories.keys())
        self.lenght = {n: len(trj) for n, trj in self.trajectories}
        self.act_trj = self.names[0]

    def load_trajectory(self, trj_p):
        data = np.load(trj_p)

        data["body_pos_w"] = data["body_pos_w"][:, self.body_ids]
        data["body_quat_w"] = data["body_quat_w"][:, self.body_ids]

        data["body_lin_vel_w"] = data["body_lin_vel_w"][:, self.body_ids]
        data["body_ang_vel_w"] = data["body_ang_vel_w"][:, self.body_ids]
        return data

    def update(self):
        if self.finished:
            return 
        self.idx += 1
        if self.idx == len(self.lenght[self.act_trj]):
            self.finished=True

    def reset(self):
        self.finished=False
        self.idx=0
    
    def switch_trajectory(self, incr):
        name = self.names.index(self.act_trj) + incr 
        self.act_trj = name

    def get_cur_slice(self, name):
        return self.trajectories[self.act_trj][name]

    @property
    def joint_pos(self):
        return self.get_cur_slice('joint_pos')[self.idx, :]
    
    @property
    def joint_vel(self):
        return self.get_cur_slice('joint_vel')[self.idx, :]
    
    @property
    def body_pos_w(self):
        return self.get_cur_slice('body_pos_w')[self.idx, :]
    
    @property
    def body_quat_w(self):
        return self.get_cur_slice('body_quat_w')[self.idx, :]

    @property
    def body_lin_vel_w(self):
        return self.get_cur_slice('body_lin_vel_w')[self.idx, :]
    
    @property
    def anchor_body_pos_w(self):
        return self.get_cur_slice('body_pos_w')[self.idx, self.anchor_id]

    @property
    def anchor_body_quat_w(self):
        return self.get_cur_slice('body_quat_w')[self.idx, self.anchor_id]

    @property
    def command(self):
        return np.concatenate([self.joint_pos, self.joint_vel], axis=1)

