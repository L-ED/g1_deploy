import sched
import time
import threading
import struct
from sshkeyboard import listen_keyboard
from loguru import logger
from termcolor import colored

from g1_deploy.rl_policy.wrappers import get_policy_wrappper
from .msg import build_interface

class Node:
    def __init__(self, robot_conf, policies_conf):
        self.robot = build_interface(robot_conf)
        self.load_policies(policies_conf)
        self.setup_ui(robot_conf)

    def load_policies(self, policies_conf):
        self.policies = {}
        self.transitions = {}
        for name, policy_conf in policies_conf:
            policy = get_policy_wrappper(policy_conf['path'], self.robot)
            self.policies[name] = (
                policy,
                {
                    frozenset(hotkey): getattr(policy, func_name)
                    for hotkey, func_name in policy_conf["func_binds"] 
                }
            )
            trns_key = frozenset(policy_conf['policy_transition_hotkey'])
            self.transitions[trns_key] = name

        self.policy, self.active_policy_funcs = self.policies['zero']
        self.active_policy_name = "zero"

    def set_policy(self, name):
        if self.active_policy_name == name:
            return
        self.active_policy_name = name
        self.policy, self.active_policy_funcs = self.policies[name]
        self.policy.reset()

    def setup_ui(self, robot_config):
        if robot_config.get("USE_JOYSTICK", False):
            print("Using joystick")
            self.use_joystick = True
            self.controller = UnitreeRemoteController()
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

        if not self.robot.updated:
            print("state not ready")
            return 
        
        self.process_user_input()
        self.prepare_hook()
        action_dict = self.policy()
        self.post_hook()
        self.robot.send_command(action_dict)

    def get_hotkey_match(self, map_dict):
        for hotkey, value in map_dict.items():
            if hotkey == self.controller.active_butns_list:
                return value
        return None

    def process_user_input(self):
        if self.use_joystick:
            self.controller.parse(self.robot.wireless_remote)
            name = self.transitions.get(self.controller.active_butns_list, None)
            if name is not None:
                self.set_policy(name)
                return
            func = self.active_policy_funcs.get(self.controller.active_butns_list, None)
            if func is not None:
                func()        

    def post_hook(self):
        if getattr(self.policy, 'finished', False) and self.pol:
            self.set_policy('walk')


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
    


class UnitreeRemoteController:
    def __init__(self):
        self.buttons_map = {
            #key, shift and data chunk index
            'R1':(0,0),
            'L1':(1,0),
            'Start':(2,0),
            'Select':(3,0),
            'R2':(4,0),
            'L2':(5,0),
            'F1':(6,0),
            'F3':(7,0),
            'A':(0,1),
            'B':(1,1),
            'X':(2,1),
            'Y':(3,1),
            'Up':(4,1),
            'Right':(5,1),
            'Down':(6,1),
            'Left':(7,1),
        }
        self.buttons_prev_state = {
            key:0 for key in self.buttons_map.keys()
        }
        self.buttons_state = {
            key:0 for key in self.buttons_map.keys()
        }
        self.active_butns_list = None
        # keys
        self.Lx = 0           
        self.Rx = 0            
        self.Ry = 0            
        self.Ly = 0

    def parse_button(self,data):
        active_butns_list = [] 
        for but_n, (idx, chunk_id) in self.buttons_map:
            btn_state = (data[chunk_id] >> idx) & 1
            prev_state = self.buttons_state[but_n]
            if btn_state > prev_state:
                active_butns_list.append(but_n + ".on_press")
            elif btn_state < prev_state:
                active_butns_list.append(but_n + ".on_release")
            elif btn_state + prev_state > 1:
                active_butns_list.append(but_n)

            self.buttons_state[but_n] = btn_state
            self.buttons_prev_state[but_n] = prev_state
            self.active_butns_list = frozenset(active_butns_list)

    def parse_stick(self,data):
        lx_offset = 4
        self.Lx = struct.unpack('<f', data[lx_offset:lx_offset + 4])[0]
        rx_offset = 8
        self.Rx = struct.unpack('<f', data[rx_offset:rx_offset + 4])[0]
        ry_offset = 12
        self.Ry = struct.unpack('<f', data[ry_offset:ry_offset + 4])[0]
        L2_offset = 16
        L2 = struct.unpack('<f', data[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        self.Ly = struct.unpack('<f', data[ly_offset:ly_offset + 4])[0]

    def parse(self,remoteData):
        self.parse_stick(remoteData)
        self.parse_button((remoteData[2],remoteData[3]))