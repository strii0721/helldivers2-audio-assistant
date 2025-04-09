from devtoolkit.Log4P import Log4P
from nn.network.CmdNetworkV2 import CmdNetwork
from utils.DatasetUtils import DatasetUtils
from nn.dataset.Preprocessing import Preprocessing
from utils.KeyboardSimulator import KeyboardSimulator

import sounddevice as sd
import torch
import torch.nn.functional as F
import time

if __name__ == "__main__":
    
    DATASET_BASE = "src/resources/dat"
    MODEL_PATH = "src/resources/model.pth"
    
    SAMPLE_RATE = 48000
    INTERVAL = 0.8
    
    THRESHOLD = 0.6
    CMD_TIMEOUT = 3
    
    CMD_DICT = {
        1: ["空爆", "↓↑↑←→"],
        2: ["火箭炮塔", "↓↑→→←"],
        3: ["机枪炮塔", "↓↑→→↑"],
        4: ["加农炮塔", "↓↑→↑←↑"],
        5: ["无后座", "↓←→→←"],
        6: ["电磁迫击炮", "↓↑→←→"],
        7: ["SOS", "↑↓→↑"],
        8: ["增援", "↑↓→←↑"],
        9: ["补给", "↓↓↑→"],
        10: ["地狱火", "↓↑←↓↑→↓↑"],
        11: ["轨道激光", "→↓↑→↓"],
        12: ["轨道汽油弹", "→→↓←→↑"],
        13: ["500千克", "↑→↓↓↓"],
    }
    
    logger = Log4P(enable_level = True,
                   enable_timestamp = True,
                   enable_source = True,
                   enable_log_file = False,
                   source = "main",)
    
    keyboardSimulator = KeyboardSimulator()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, category_number = DatasetUtils.get_dataframe_distributed(DATASET_BASE)
    model = CmdNetwork(category_number = category_number).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    logger.info("🎙️  开始监听麦克风...")
    command_start = False
    while True:
        audio = sd.rec(int(SAMPLE_RATE * INTERVAL), samplerate = SAMPLE_RATE, channels = 2)
        sd.wait()
        audio = torch.tensor(audio.T, dtype=torch.float32) # [1, num_samples]

        audio = Preprocessing.isometricalization(audio = audio,
                                                 sample_rate = SAMPLE_RATE,
                                                 length = INTERVAL)
        audio = Preprocessing.mono(audio = audio)
        audio = Preprocessing.mel_spectrogram(audio = audio,
                                              sample_rate = SAMPLE_RATE)
        audio = audio.unsqueeze(0)   
        with torch.no_grad():
            audio = audio.to(device) 
            output = model(audio)
            probs = F.softmax(output, dim=1)           # 概率分布
            max_prob, pred_index = torch.max(probs, 1) # 获取最大概率及其索引
            max_prob = max_prob.item()
            pred_index = pred_index.item()
            if max_prob >= THRESHOLD and pred_index != 0:
                if pred_index == 14:
                     keyboardSimulator.start()
                     command_start = True
                     logger.info(f"✅  指令：")
                     ticker = time.time()
                else:
                    if command_start:
                        command_name = CMD_DICT[pred_index][0]
                        command_sequence = CMD_DICT[pred_index][1]
                        keyboardSimulator.read_cmd_seq(command_sequence)
                        keyboardSimulator.end()
                        command_start = False
                        logger.info(f"▶️  {command_name}：{command_sequence}")
            else:
                if command_start and time.time() - ticker > CMD_TIMEOUT:
                    keyboardSimulator.end()
                    command_start = False
                    logger.info(f"❌  指令被取消")
