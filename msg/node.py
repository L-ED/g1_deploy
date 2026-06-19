import sched
import time
import threading
from sshkeyboard import listen_keyboard
from loguru import logger
from termcolor import colored

from g1_deploy.rl_policy.wrappers import get_policy_wrappper
from .zmq.interface import build_interface

class Node:
    def __init__(self, robot_conf, policy_conf):
        self.robot = build_interface(robot_conf)
        self.policy = get_policy_wrappper(policy_conf)
        self.setup_ui(robot_conf)

    def setup_ui(self, robot_config):
        if robot_config.get("USE_JOYSTICK", False):
            print("Using joystick")
            self.use_joystick = True
            self.wc_msg = None  # type: ignore
            self.last_wc_msg = self.robot.read_wireless_controller()
            print("Wireless Controller Initialized")
        else:
            print("Using keyboard")
            self.use_joystick = False
            self.key_listener_thread = threading.Thread(
                target=self.start_key_listener, daemon=True
            )
            self.key_listener_thread.start()
        
    def run(self):
        try:
            scheduler = sched.scheduler(time.perf_counter, time.sleep)
            next_run_time = time.perf_counter()
            
            while True:
                scheduler.enterabs(next_run_time, 1, self._rl_step_scheduled, ())
                scheduler.run()
                
                next_run_time += self.rl_dt
                self.total_inference_cnt += 1

                if self.total_inference_cnt % 100 == 0:
                    self.perf_dict = {}
        except KeyboardInterrupt:
            pass

    def _rl_step_scheduled(self):

        if not self.robot.state_ready:
            print("state not ready")
            return 
        
        self.process_user_input()
        self.prepare_hook()
        action_dict = self.policy(self.robot.state)
        self.post_hook()
        self.robot.send_command(action_dict)

    def process_user_input(self):
        if self.use_joystick:
            self.process_joystick_input()


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
    