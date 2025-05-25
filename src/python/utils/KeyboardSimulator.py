from pynput.keyboard import Key, Controller
from pynput.mouse import Button
from pynput.mouse import Controller as MouseController
import time

class KeyboardSimulator:
    
    KEY_PRESS_INTERVAL = 0.05
    
    def __init__(self):
        self.controller = Controller()
        self.mouse_controller =  MouseController()
    
    def read_cmd_seq(self,cmd_seq):
        for cmd in cmd_seq:
            match cmd:
                case "↑":
                    self.controller.press(Key.up)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                    self.controller.release(Key.up)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                case "↓":
                    self.controller.press(Key.down)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                    self.controller.release(Key.down)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                case "←":
                    self.controller.press(Key.left)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                    self.controller.release(Key.left)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                case "→":
                    self.controller.press(Key.right)
                    time.sleep(self.KEY_PRESS_INTERVAL)
                    self.controller.release(Key.right)
                    time.sleep(self.KEY_PRESS_INTERVAL)
    
    def start(self):
        self.controller.press(Key.ctrl)
        # self.mouse_controller.press(Button.button4)
        time.sleep(self.KEY_PRESS_INTERVAL)
    
    def end(self):
        self.controller.release(Key.ctrl)
        # self.mouse_controller.release(Button.button4)
        time.sleep(self.KEY_PRESS_INTERVAL)