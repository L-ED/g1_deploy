
import inspect
import numpy as np
from . import obs_funcs as obs
from .circular_buffer import CircularBuffer

OBS_FUNC_REGISTRY = dict(inspect.getmembers(obs, inspect.isfunction))

class ObsManager:
    def __init__(
        self,
        env,
        observation_config
    ):
        self._env = env
        self.observation_config = observation_config
        self.group_history_buffer = {}
        for group_name, group_cfg in observation_config['terms']:
            group_delay_bufs = {}
            # if group_cfge
            for term_cfg in group_cfg:
                term_cfg['func'] = OBS_FUNC_REGISTRY[term_cfg['func_name']]
                if term_cfg['history_length'] > 0:
                    group_delay_bufs[term_cfg['name']] =  CircularBuffer(
                        max_len=term_cfg['history_length'],
                        batch_size=self._env.num_envs
                    )
            self.group_history_buffer[group_name] = group_delay_bufs
        
    def compute(self) -> np.ndarray:
        groups_obs = {}
        for group_name, group_cfg in self.observation_config[group_name]:
            groups_obs[group_name] = self._compute_group(group_name, group_cfg)
        return groups_obs
    
    def _compute_group(self, group_name, group_cfg) -> np.ndarray:
        # update only if outdated
        group_obs = {}
        for term_name, term_cfg in group_cfg['terms']:
            obs = term_cfg['func'](self._env, **term_cfg['params']).clone()
        
        if term_cfg['clip']:
            obs = obs.clip(min=term_cfg['clip'][0], max=term_cfg['clip'][1])
        if term_cfg['scale'] is not None:
            scale = term_cfg['scale']
            obs *= scale

        if term_cfg['history_length'] > 0:
            circular_buffer = self.group_history_buffer[group_name][term_name]
            if self.update_history or not circular_buffer.is_initialized:
                circular_buffer.append(obs)

            if term_cfg['flatten_history_dim']:
                group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
            else:
                group_obs[term_name] = circular_buffer.buffer
        else:
            group_obs[term_name] = obs

        if group_cfg['concatenate_terms']:
            result = np.concatenate(
                list(group_obs.values()), axis=self.group_cfg['concatenate_dim']
            )
            return result
        return group_obs
    
    def reset(self):
        for obs_gr_bufs in self.group_history_buffer.values():
            for buf in obs_gr_bufs:
                buf.reset()

    def update(self):
        pass