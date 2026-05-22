from rl_policy.utils.state_processor import StateProcessor
from rl_policy.utils.command_sender import CommandSender

class ZMQInterface:

    def __init__(self, robot_config):
        
        self.robot_config = robot_config
        robot_type = robot_config["ROBOT_TYPE"]
        if robot_type == "g1_real":
            # example: sys.path.append("/home/unitree/User/unitree_sdk2/build/lib")
            sys.path.append("/path/to/your/unitree_sdk2/build/lib")
            import g1_interface
            network_interface = robot_config.get("INTERFACE", None)
            self.robot = g1_interface.G1Interface(network_interface)
            try:
                self.robot.set_control_mode(g1_interface.ControlMode.PR)
            except Exception:
                pass  # Ignore if firmware already in the correct mode
            robot_config["robot"] = self.robot

    def init_low_cmd(self):
        # init low cmd publisher
        if self.robot_type != "g1_real":
            self.zmq_context = zmq.Context.instance()
            self.low_cmd_port = robot_config.get(
                "LOW_CMD_PORT", PORTS.get("low_cmd", 55901)
            )
            bind_addr = robot_config.get("LOW_CMD_BIND_ADDR", "*")
            bind_endpoint = f"tcp://{bind_addr}:{self.low_cmd_port}"

            self.lowcmd_socket: zmq.Socket = self.zmq_context.socket(zmq.PUB)
            self.lowcmd_socket.setsockopt(zmq.SNDHWM, 1)
            self.lowcmd_socket.setsockopt(zmq.LINGER, 0)
            self.lowcmd_socket.bind(bind_endpoint)
            # Give subscribers time to connect before sending commands
            time.sleep(0.1)
        else:
            self.lowcmd_socket = None

        self.InitLowCmd()


    def init_low_state(self):
        self.robot_type = robot_config["ROBOT_TYPE"]
        self.mocap_ip = robot_config.get("MOCAP_IP", "localhost")
        # Initialize state source
        if self.robot_type == "g1_real":
            self.robot = robot_config["robot"]
        else:
            supported_types = {
                "h1",
                "go2",
                "g1_29dof",
                "h1-2_27dof",
                "h1-2_21dof",
            }
            if self.robot_type not in supported_types:
                raise NotImplementedError(
                    f"Robot type {self.robot_type} is not supported"
                )

            self.low_state_port = robot_config.get(
                "LOW_STATE_PORT", PORTS.get("low_state", 55900)
            )
            state_host = robot_config.get("LOW_STATE_HOST", "127.0.0.1")
            state_endpoint = f"tcp://{state_host}:{self.low_state_port}"

            self.zmq_context = zmq.Context.instance()
            self.low_state_socket: zmq.Socket = self.zmq_context.socket(zmq.SUB)
            self.low_state_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self.low_state_socket.setsockopt(zmq.CONFLATE, 1)
            self.low_state_socket.setsockopt(zmq.RCVTIMEO, 10)
            self.low_state_socket.connect(state_endpoint)
            self.latest_low_state: LowStateMessage | None = None

    def register_subscriber(self, object_name: str, port: int | None = None):
        if object_name in self.mocap_subscribers:
            return
        # init ZMQ subscriber
        port = PORTS.get(f"{object_name}_pose", port)
        subscriber = ZMQSubscriber(port)
        self.mocap_subscribers[object_name] = subscriber

        def _sub_thread(obj_name: str):
            while True:
                try:
                    pose_msg = self.mocap_subscribers[obj_name].receive_pose()
                    if pose_msg:
                        with self.mocap_data_lock:
                            self.mocap_data[f"{obj_name}_pos"] = pose_msg.position
                            self.mocap_data[f"{obj_name}_quat"] = pose_msg.quaternion
                except zmq.Again:
                    time.sleep(0.001)
                except Exception as e:
                    logger.warning(f"{obj_name} subscriber error: {e}")
                    time.sleep(0.01)

        # start subscriber thread
        th = threading.Thread(target=_sub_thread, args=(object_name,), daemon=True)
        th.start()
        self.mocap_threads[object_name] = th

    def _prepare_low_state(self):
        if hasattr(self, "low_state_socket"):
            self._receive_low_state()
            if not self.latest_low_state:
                return False

            low_state = self.latest_low_state
            self.root_quat_b[:] = low_state.quaternion
            self.root_ang_vel_b[:] = low_state.gyroscope

            source_joint_pos = low_state.joint_positions
            source_joint_vel = low_state.joint_velocities
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = source_joint_pos[src_idx]
                self.joint_vel[dst_idx] = source_joint_vel[src_idx]

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
            self.root_quat_b[:] = state.imu.quat  # [w, x, y, z]
            self.root_ang_vel_b[:] = state.imu.omega

            # Joints
            for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
                self.joint_pos[dst_idx] = state.motor.q[src_idx]
                self.joint_vel[dst_idx] = state.motor.dq[src_idx]
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


    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        if self.robot_type != "g1_real":
            self.cmd_q[self.joint_indices_unitree] = cmd_q
            self.cmd_dq[self.joint_indices_unitree] = cmd_dq
            self.cmd_tau[self.joint_indices_unitree] = cmd_tau
            
            message = LowCmdMessage(
                q_target=self.cmd_q,
                dq_target=self.cmd_dq,
                tau_ff=self.cmd_tau,
                kp=self.joint_kp_unitree,
                kd=self.joint_kd_unitree,
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

            q_target = list(cmd.q_target)
            dq_target = list(cmd.dq_target)
            tau_ff = list(cmd.tau_ff)
            kp = list(cmd.kp)
            kd = list(cmd.kd)
            for i_policy, idx_unitree in enumerate(self.joint_indices_unitree):
                q_target[idx_unitree] = float(cmd_q[i_policy])
                dq_target[idx_unitree] = float(cmd_dq[i_policy])
                tau_ff[idx_unitree] = float(cmd_tau[i_policy])
                kp[idx_unitree] = float(kp_scaled[idx_unitree])
                kd[idx_unitree] = float(kd_scaled[idx_unitree])

            cmd.q_target = q_target
            cmd.dq_target = dq_target
            cmd.tau_ff = tau_ff
            cmd.kp = kp
            cmd.kd = kd

            self.robot.write_low_command(cmd)


class Node:

    def __init__(self):

        # ------------------------------------------------------
        # Joystick / keyboard setup (mirrors base_policy logic)
        # ------------------------------------------------------
        if robot_config.get("USE_JOYSTICK", False):
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None  # type: ignore
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        else:
            import threading
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()

        # Plug-in our custom state processor & command sender
        self.state_processor = StateProcessor(robot_config, policy_config["isaac_joint_names"])
        self.command_sender = CommandSender(robot_config, policy_config)


    def process_joystick_input(self):
        """Poll current wireless controller state and translate to high-level key events."""
        try:
            self.wc_msg = self.robot.read_wireless_controller()
        except Exception:
            return

        if self.wc_msg is None:
            return

        # print(f"wc_msg.A: {self.wc_msg.A}")
        if self.wc_msg.A and not self.last_wc_msg.A:
            self.handle_joystick_button("A")
        if self.wc_msg.B and not self.last_wc_msg.B:
            self.handle_joystick_button("B")
        if self.wc_msg.X and not self.last_wc_msg.X:
            self.handle_joystick_button("X")
        if self.wc_msg.Y and not self.last_wc_msg.Y:
            self.handle_joystick_button("Y")
        if self.wc_msg.L1 and not self.last_wc_msg.L1:
            self.handle_joystick_button("L1")
        if self.wc_msg.L2 and not self.last_wc_msg.L2:
            self.handle_joystick_button("L2")
        if self.wc_msg.R1 and not self.last_wc_msg.R1:
            self.handle_joystick_button("R1")
        if self.wc_msg.R2 and not self.last_wc_msg.R2:
            self.handle_joystick_button("R2")
        
        self.last_wc_msg = self.wc_msg


    def handle_joystick_button(self, cur_key):
        if cur_key == "R1":
            logger.info("Using policy actions")
            self.use_policy_action = True
            self.get_ready_state = False    
            if self.task_type == "reward":
                total_z = self.num_selected_z
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            if self.task_type == "goal":
                logger.info(colored(f"Switch to goal={list(self.z_dict.keys())[self.z_index]} (Count: {self.z_index+1}/{len(self.z_dict)})", "blue"))
            if self.task_type == "tracking":
                self.t = self.t_stop

        elif cur_key == "R2":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info(colored("Actions set to zero", "blue"))
        elif cur_key == "A":
            self.get_ready_state = True
            self.init_count = 0
            logger.info(colored("Setting to init state (do this when robot was in a bad shape)", "blue"))
        elif cur_key == "B":
            if self.task_type == "tracking":
                logger.info("Starting motion")
                self.start_motion = True
                self.t = self.t_start
            else:
                logger.info(colored(f"Commmand [ is undefined in current task type {self.task_type}!", "red"))
                pass
        elif cur_key == "X":
            self.z_index = 0
            self.start_motion = False
            if self.task_type == "tracking":
                self.t = self.t_stop
            logger.info("Resetting to stop state")
        elif cur_key == "Y":
            if self.task_type == "reward":
                if self.z_index >= self.num_selected_z - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                total_z = self.num_selected_z if self.task_type.startswith("reward-multiple-z-selection-duplicate") else self.num_selected_rewards
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            elif self.task_type == "goal":
                if self.z_index >= self.num_selected_goals - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                logger.info(colored(f"Switch to goal {list(self.z_dict.keys())[self.z_index]} ({self.z_index+1}/{self.num_selected_goals})", "blue"))

        # Debug print for kp level tuning
        if cur_key in ["Y+left", "Y+right", "A+left", "A+right"]:
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))


    # ----------------------------- Keyboard handling -----------------------------
    def start_key_listener(self):
        """Start a key listener using sshkeyboard (same as BasePolicy)."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError as e:
                logger.warning(f"Keyboard key {keycode}. Error: {e}")

        listener = listen_keyboard(on_press=on_press)
        listener.start()
        listener.join()


    def handle_keyboard_button(self, keycode):
        if keycode == "]":
            logger.info("Using policy actions")
            self.use_policy_action = True
            self.get_ready_state = False    
            if self.task_type == "reward":
                total_z = self.num_selected_z
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            if self.task_type == "goal":
                logger.info(colored(f"Switch to goal={list(self.z_dict.keys())[self.z_index]} (Count: {self.z_index+1}/{len(self.z_dict)})", "blue"))
            if self.task_type == "tracking":
                self.t = self.t_stop
        elif keycode == "[":
            if self.task_type == "tracking":
                logger.info("Starting motion")
                self.start_motion = True
                self.t = self.t_start
            else:
                logger.info(colored(f"Commmand [ is undefined in current task type {self.task_type}!", "red"))
                pass
        elif keycode == "n":
            if self.task_type == "reward":
                if self.z_index >= self.num_selected_z - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                total_z = self.num_selected_z if self.task_type.startswith("reward-multiple-z-selection-duplicate") else self.num_selected_rewards
                logger.info(colored(f"Switch to reward={self.selected_z_names[self.z_index]} (Count: {self.z_index+1}/{total_z})", "blue"))
            elif self.task_type == "goal":
                if self.z_index >= self.num_selected_goals - 1:
                    self.z_index = 0
                else:
                    self.z_index += 1
                logger.info(colored(f"Switch to goal {list(self.z_dict.keys())[self.z_index]} ({self.z_index+1}/{self.num_selected_goals})", "blue"))
        elif keycode == "p":
            self.z_index = 0
            self.start_motion = False
            if self.task_type == "tracking":
                self.t = self.t_stop
            logger.info("Resetting to stop state")
        elif keycode == "o":
            self.use_policy_action = False
            self.get_ready_state = False
            logger.info("Actions set to zero")
        elif keycode == "i":
            self.get_ready_state = True
            self.init_count = 0
            logger.info("Setting to init state")
        elif keycode == "w":
            self.lin_vel_command[0, 0] += 0.1
        elif keycode == "s":
            self.lin_vel_command[0, 0] -= 0.1
        elif keycode == "a":
            self.lin_vel_command[0, 1] += 0.1
        elif keycode == "d":
            self.lin_vel_command[0, 1] -= 0.1
        elif keycode == "q":
            self.ang_vel_command[0, 0] -= 0.1
        elif keycode == "e":
            self.ang_vel_command[0, 0] += 0.1
        elif keycode == "z":
            self.ang_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
        elif keycode == "5":
            self.command_sender.kp_level -= 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "6":
            self.command_sender.kp_level += 0.01
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "4":
            self.command_sender.kp_level -= 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "7":
            self.command_sender.kp_level += 0.1
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
        elif keycode == "0":
            self.command_sender.kp_level = 1.0
            for i in range(len(self.command_sender.robot_kp)):
                self.command_sender.robot_kp[i] = self.robot.MOTOR_KP[i] * self.command_sender.kp_level
            logger.info(colored(f"Debug kp level: {self.command_sender.kp_level}", "green"))
            logger.info(colored(f"Debug kp: {self.command_sender.robot_kp}", "green"))
    