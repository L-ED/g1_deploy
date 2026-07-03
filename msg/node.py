import sched
import time
import threading
import struct
import yaml
import os
from sshkeyboard import listen_keyboard
from loguru import logger
from termcolor import colored
from pathlib import Path
from g1_deploy.rl_policy.wrappers import get_policy_wrappper
from .interface import INTERFACE_MAP
from g1_deploy.utils import load_yaml

class Node:
    def __init__(self, conf_path):
        robot_conf = load_yaml(os.path.join(conf_path, 'robot.yaml'))
        policies_conf = load_yaml(os.path.join(conf_path, 'policy.yaml'))
        self.robot = INTERFACE_MAP[robot_conf['con_type']](robot_conf)
        self.load_policies(policies_conf)
        self.setup_ui(robot_conf)
        self.robot.dump()

    def load_policies(self, policies_conf):
        self.policies = {}
        self.transitions = {}
        for name, policy_conf in policies_conf:
            policy = get_policy_wrappper(policy_conf['path'], self.robot)
            pol_funcs = {}
            for hotkey, func_name in policy_conf["func_binds"]:
                src = policy if hasattr(policy, func_name) else self
                func = getattr(src, func_name) 
                if func is None:
                    raise ValueError(f"no func {func_name}")
                pol_funcs[frozenset(hotkey)].append()

            self.policies[name] = (policy, pol_funcs)                
            trns_key = frozenset(policy_conf['policy_transition_hotkey'])
            self.transitions[trns_key] = name

        self.policy, self.active_policy_funcs = self.policies['zero']
        self.active_policy_name = None
        self.active_policy_funcs = None

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

    def prepare_hook(self):
        if self.active_policy_name is not None:
            self.robot.activate()

    def dump(self):
        self.robot.dump()
        self.active_policy_name=None
        self.active_policy_funcs = None


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
        # for one traj policy
        if getattr(self.policy, 'finished', False):
            self.set_policy('walk')

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
        pass
    


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

    def parse(self, remoteData):
        self.parse_stick(remoteData)
        self.parse_button((remoteData[2],remoteData[3]))


if __name__ == "__main__":

    file_path = Path(__file__).resolve()
    dir_path = file_path.parent.parent
    config_path = os.path.join(file_path, 'config', 'policy')
    node = Node(config_path)
    node.run()



    