import inspect
import numpy as np
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from msg.zmq.state_processor import StateProcessor
    from rl_policy.base_policy import BasePolicy

class _RegistryMixin:
    
    def __init_subclass__(cls) -> None:
        """Put the subclass in the global registry"""
        if not hasattr(cls, 'registry'):
            cls.registry = {}
            
        cls_name = cls.__name__
        cls._file = inspect.getfile(cls)
        cls._line = inspect.getsourcelines(cls)[1]
        if cls_name not in cls.registry:
            cls.registry[cls_name] = cls
        else:
            conflicting_cls = cls.registry[cls_name]
            location = f"{conflicting_cls._file}:{conflicting_cls._line}"
            raise ValueError(f"Term {cls_name} already registered in {location}")

class Observation(_RegistryMixin):
    def __init__(self, env: "BasePolicy", scale: float = 1.0, **kwargs):
        self.env = env
        self.state_processor = env.robot.state
        self.scale = scale
    
    def reset(self):
        pass

    def update(self, data: Dict[str, Any]) -> None:
        pass

    def compute(self, scale: float = 1.0) -> np.ndarray:
        raise NotImplementedError

class ObsManager:
    def __init__(
        self,
        env,
        observation_config
    ):
        self.groups_delay_buffers = {}
        for group_name, group_cfg in observation_config:
            group_delay_bufs = {}
            # if group_cfge
            for term_cfg in group_cfg:
                term_cfg['func'] = OBS_FUNC_REGISTRY[term_cfg['func_name']]
                if term_cfg.term_cfg.history_length > 0:
                    group_delay_bufs[term_cfg['name']] =  CircularBuffer(
                        max_len=term_cfg['history_length'],
                        batch_size=self._env.num_envs
                    )
            self.groups_delay_buffers[group_name] = group_delay_bufs
                

    def compute(self) -> np.ndarray:
        output = self._compute()
        return output
    
    def _compute_group(self) -> np.ndarray:
        # update only if outdated
        for term_name, term_cfg in obs_terms:
        obs = term_cfg.func(self._env, **term_cfg.params).clone()
        
        if term_cfg.clip:
            obs = obs.clip_(min=term_cfg.clip[0], max=term_cfg.clip[1])
        if term_cfg.scale is not None:
            scale = term_cfg.scale
            obs = obs.mul_(scale)

        if term_cfg.history_length > 0:
            circular_buffer = self._group_obs_term_history_buffer[group_name][term_name]
            if update_history or not circular_buffer.is_initialized:
            circular_buffer.append(obs)

            if term_cfg.flatten_history_dim:
            group_obs[term_name] = circular_buffer.buffer.reshape(self._env.num_envs, -1)
            else:
            group_obs[term_name] = circular_buffer.buffer
        else:
            group_obs[term_name] = obs

        if self._group_obs_concatenate[group_name]:
            result = torch.cat(
                list(group_obs.values()), dim=self._group_obs_concatenate_dim[group_name]
            )