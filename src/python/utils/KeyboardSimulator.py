from pynput.keyboard import Key, Controller
import time

class KeyboardSimulator:
    
    def __init__(self):
        self.controller = Controller()
    
    def read_cmd_seq(self,cmd_seq):
        for cmd in cmd_seq:
            match cmd:
                case "↑":
                    self.controller.press(Key.up)
                    self.controller.release(Key.up)
                    time.sleep(0.01)
                case "↓":
                    self.controller.press(Key.down)
                    self.controller.release(Key.down)
                    time.sleep(0.01)
                case "←":
                    self.controller.press(Key.left)
                    self.controller.release(Key.left)
                    time.sleep(0.01)
                case "→":
                    self.controller.press(Key.right)
                    self.controller.release(Key.right)
                    time.sleep(0.01)
    
    def start(self):
        self.controller.press(Key.ctrl)
        time.sleep(0.05)
    
    def end(self):
        self.controller.release(Key.ctrl)
        time.sleep(0.05)