
from dataclasses import dataclass, field
import numpy as np

@dataclass
class State:
    ndof: int  # User provides this during initialization
    joint_pos: np.ndarray = field(init=False)
    joint_vel: np.ndarray = field(init=False)
    joint_tau: np.ndarray = field(init=False)
    root_quat_b: np.ndarray = field(default_factory=lambda: np.zeros(4))
    root_lin_acc_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    root_ang_vel_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    def __post_init__(self):
        # Generate the zero array based on the dynamic size argument
        self.joint_pos = np.zeros(self.size)
        self.joint_vel = np.zeros(self.size)
        self.joint_tau = np.zeros(self.size)