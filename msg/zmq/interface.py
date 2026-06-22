import zmq
import sys
from utils.strings import resolve_matching_names_values
from utils.strings import unitree_joint_names
from msg.zmq.common import LowCmdMessage, LowStateMessage, PORTS
from dataclasses import dataclass
import numpy as np
from g1_deploy.msg.utils import State

class ZMQInterface:

    def __init__(self, robot_config, policy_joint_order):
        
        self.robot_config = robot_config
        self.state = None
        self.robot_type = robot_config["ROBOT_TYPE"]
        self.unitree_index = [
            policy_joint_order.index(name) for name in 
            robot_config['robot_joint_names']]
        self.state = State(len(self.unitree_index))
        self.default_pos = np.array(robot_config['default_pos'])
        self.cmd_q = np.zeros_like(self.default_pos)
        self.cmd_dq = np.zeros_like(self.default_pos)
        self.cmd_tau = np.zeros_like(self.default_pos)
        self.init_connection()

    def init_connection(self):
        if self.robot_type == "g1_real":
            # example: sys.path.append("/home/unitree/User/unitree_sdk2/build/lib")
            sys.path.append("/path/to/your/unitree_sdk2/build/lib")
            import g1_interface
            network_interface = self.robot_config.get("INTERFACE", None)
            self.robot = g1_interface.G1Interface(network_interface)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception:
                pass  # Ignore if firmware already in the correct mode
            self.robot_config["robot"] = self.robot
            self.lowcmd_socket = None

        else:
            # cmd
            self.zmq_context = zmq.Context.instance()
            self.low_cmd_port = self.robot_config.get(
                "LOW_CMD_PORT", PORTS.get("low_cmd", 55901)
            )
            bind_addr = self.robot_config.get("LOW_CMD_BIND_ADDR", "*")
            bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

            self.lowcmd_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
            self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
            self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
            self.lowcmd_socket.bind(bind_endpoint)
            # Give subscribers time to connect before sending commands
            time.sleep(0.1)


            self.low_state_port = self.robot_config.get(
                "LOW_STATE_PORT", PORTS.get("low_state", 55900)
            )
            state_host = self.robot_config.get("LOW_STATE_HOST", "127.0.0.1")
            state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

            self.zmq_context = zmq.Context.instance()
            self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
            self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
            self.low_state_socket.connect(state_endpoint)
            self.latest_low_state: LowStateMessage | None = None


    def receive_lowstate(self):
        if hasattr(self, "low_state_socket"):
            self._receive_low_state()
            if not self.latest_low_state:
                return False

            low_state = self.latest_low_state
            self.state.base_ang_vel[:] = low_state.gyroscope
            self.state.quat[:] = low_state.quaternion  # [w, x, y, z]
            # self.state.base_lin_acc[:] = state.imu.omega

            source_joint_pos = low_state.joint_positions
            source_joint_vel = low_state.joint_velocities
            for dst_idx, src_idx in enumerate(self.unitree_index):
                self.state.joint_pos[dst_idx] = source_joint_pos[src_idx]
                self.state.joint_vel[dst_idx] = source_joint_vel[src_idx]

            return True
        elif hasattr(self, "robot"):
            try:
                state = self.robot.read_low_state()
            except Exception as e:
                logger.warning(f"Failed to read G1 low state: {e}")
                return False

            if state is None:
                return False

            # IMU
            self.state.base_ang_vel[:] = state.imu.omega
            self.state.quat[:] = state.imu.quat  # [w, x, y, z]
            self.state.base_lin_acc[:] = state.imu.omega

            # Joints
            for dst_idx, src_idx in enumerate(self.unitree_index):
                self.state.joint_pos[dst_idx] = state.motor.q[src_idx] 
                self.state.joint_vel[dst_idx] = state.motor.dq[src_idx] 
            return True

    def _receive_low_state(self):
        """Fetch the most recent low state message from the ZMQ socket."""
        if not hasattr(self, "low_state_socket"):
            return

        while True:
            try:
                data = self.low_state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                self.latest_low_state = LowStateMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low state message: {exc}")


    def send_command(self, cmd_dict: dict):
        self.cmd_q[self.joint_indices_unitree] = cmd_dict['q']
        self.cmd_dq[self.joint_indices_unitree] = cmd_dict['dq']
        self.cmd_tau[self.joint_indices_unitree] = cmd_dict['tau']
        self.kp[self.joint_indices_unitree] = cmd_dict['tau']
        self.kd[self.joint_indices_unitree] = cmd_dict['tau']

        if self.robot_type != "g1_real":
            message = LowCmdMessage(
                q_target=self.cmd_q,
                dq_target=self.cmd_dq,
                tau_ff=self.cmd_tau,
                kp=self.kp,
                kd=self.kd,
            )
            try:
                self.lowcmd_socket.send(message.to_bytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
        else:
            cmd = self.robot.create_zero_command()

            # Apply kp_level scaling (kd remains constant, consistent with original implementation)
            kp_scaled = self.joint_kp_unitree * self._kp_level
            kd_scaled = self.joint_kd_unitree

            cmd.q_target = self.cmd_q.tolist()
            cmd.dq_target = self.cmd_dq.tolist()
            cmd.tau_ff = self.cmd_tau.tolist()
            cmd.kp = self.kp.tolist()
            cmd.kd = self.kd.tolist()

            self.robot.write_low_command(cmd)
