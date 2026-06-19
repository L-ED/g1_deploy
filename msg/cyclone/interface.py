import time
import sys

import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from g1_deploy.msg.utils import State

class CDDSInterface:

    def __init__(self, robot_cfg):
         # create publisher #
        self.robot_cfg = robot_cfg
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.receive_lowstate, 10)
        self.unitree_index = self.robot_cfg['unitree_index']
        # https://support.unitree.com/home/en/G1_developer/basic_motion_routine
        self.motor_mode_pr = 0 #  1 is closed chain, 0 is open
        self.update_mode_machine_ = False
        self.state = State()

    def send_command(self, cmd_dict):
        for i in range(self.unitree_index):
            idx = self.unitree_index[i]
            self.low_cmd.mode_pr = self.motor_mode_pr
            self.low_cmd.mode_machine = self.mode_machine_
            self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
            self.low_cmd.motor_cmd[i].q = cmd_dict['q'][idx] 
            self.low_cmd.motor_cmd[i].dq = cmd_dict['dq'][idx]
            self.low_cmd.motor_cmd[i].kp = cmd_dict['kp'][idx] 
            self.low_cmd.motor_cmd[i].kd = cmd_dict['kd'][idx]
            self.low_cmd.motor_cmd[i].tau = cmd_dict['tau'][idx]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
    
    def receive_lowstate(self, msg: LowState_):
        self.low_state = msg

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True

        self.state.quat[:] = np.array(
            msg.imu_state.quaternion, dtype=np.float32)  # scalar first
        self.state.base_lin_acc[:] = np.array(msg.imu_state.accelerometer, dtype=np.float32)
        self.state.base_ang_vel[:] = np.array(msg.imu_state.gyroscope, dtype=np.float32)

        # g1 has no unused sections, so it works
        for i, state in enumerate(msg.motor_state):
            idx = self.unitree_index[i]
            self.state.joint_pos[idx] = state.q
            self.state.joint_vel[idx] = state.dq
            self.state.joint_tau[idx] = state.tau_est

        self.quaternion = np.array(
            msg.imu_state.quaternion, dtype=np.float32
        )  # scalar first
        self.acc = np.array(msg.imu_state.accelerometer, dtype=np.float32)
        self.gyro = np.array(msg.imu_state.gyroscope, dtype=np.float32)

    def update_sim_state(self, msg):
        # get proprioception from simulator
        self.sim_base_position = np.array(msg.position)  # world frame
        self.sim_base_lin_vel = np.array(msg.velocity)  # base frame
