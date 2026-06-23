
import os
import onnxruntime
from rl_policy.observations import Observation, ObsGroup
import numpy as np
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



class PolicyWrapper:

    def __init__(self, policy_dir_path, robot, device='cpu'):
        policy_path = os.path.join(policy_dir_path, 'policy.onnx')
        config_path = os.path.join(policy_dir_path, 'config.yaml')
        self.device = device
        self.robot = robot
        self.parse_config(config_path)
        self.setup_policy(policy_path)

    def parse_config(self):

        self.joint_order = self.cfg["joint_order"]
        self.num_dof = len(self.joint_order)
        self.default_dof_angles = self.to_tensor(
            self.cfg["init_state"]["default_joint_pos"], self.joint_order)
        self.last_action  = np.zeros(len(self.joint_order))
        self.action_manager = ActionManager(self, self.cfg['control'])
        self.prev_action = np.zeros(len(self.joint_order))
        self.setup_observations()

    def to_tensor(self, cfg_dict, order, dtype = np.float32):
        return np.asarray(
            [cfg_dict[name] for name in order],
            dtype=dtype)

    def setup_policy(self, policy_path):
        # load onnx policy
        logger.info(f"Loading onnx policy from {policy_path}")
        self.onnx_policy_session = onnxruntime.InferenceSession(policy_path)
        self.onnx_input_name = self.onnx_policy_session.get_inputs()[0].name
        self.onnx_output_name = self.onnx_policy_session.get_outputs()[0].name
        def policy_act(obs):
            return self.onnx_policy_session.run([self.onnx_output_name], {self.onnx_input_name: obs})[0]
        self.policy = policy_act


    def setup_observations(self):
        """Setup observations for policy inference"""
        self.observations: Dict[str, ObsGroup] = {}
        self.reset_callbacks = []
        self.update_callbacks = []
        # Create observation instances based on config
        for obs_group, obs_items in self.obs_cfg.items():
            print(f"obs_group: {obs_group}")
            obs_funcs = {}
            for obs_name, obs_config in obs_items.items():
                obs_class: Type[Observation] = Observation.registry[obs_name]
                obs_func = obs_class(env=self, **obs_config)
                obs_funcs[obs_name] = obs_func
                self.reset_callbacks.append(obs_func.reset)
                self.update_callbacks.append(obs_func.update)
                print(f"\t{obs_name}: {obs_config}")
            self.observations[obs_group] = ObsGroup(obs_group, obs_funcs)

    def reset(self):
        for reset_callback in self.reset_callbacks:
            reset_callback()

    def update(self):
        for update_callback in self.update_callbacks:
            update_callback()

    def prepare_obs_for_rl(self):
        """Prepare observation for policy inference using observation classes"""
        obs_dict: Dict[str, np.ndarray] = {}
        self.update()
        for obs_group in self.observations.values():
            obs = obs_group.compute()
            obs_dict[obs_group.name] = obs[None, :].astype(np.float32)
        obs = obs_dict[obs_group.name]

        return obs_dict, obs

    def __call__(self):
        obs_d, obs = self.prepare_obs_for_rl()
        raw_action = self.policy(obs)
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
