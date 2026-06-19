
from dataclasses import dataclass, field
import numpy as np

@dataclass
class State:
    ndof: int  # User provides this during initialization
    joint_pos: np.ndarray = field(init=False)
    joint_vel: np.ndarray = field(init=False)
    joint_tau: np.ndarray = field(init=False)
    quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    base_lin_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    def __post_init__(self):
        # Generate the zero array based on the dynamic size argument
        self.joint_pos = np.zeros(self.size)
        self.joint_vel = np.zeros(self.size)
        self.joint_tau = np.zeros(self.size)