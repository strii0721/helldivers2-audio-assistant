from devtoolkit.Log4P import Log4P
from nn.network.CmdNetworkV2 import CmdNetwork
from utils.DatasetUtils import DatasetUtils
from nn.dataset.Preprocessing import Preprocessing
from utils.KeyboardSimulator import KeyboardSimulator

import sounddevice as sd
import torch
import torch.nn.functional as F
import time
import csv
import soundfile as sf
import os

if __name__ == "__main__":
    
    DATASET_BASE = "src/resources/dat"
    MODEL_PATH = "src/resources/model.pth"
    DICT_PATH = "src/resources/cmd-dict.csv"
    AUDIO_SAVE_BASE = "src/resources/dat"
    AUDIO_EXT = "wav"
    
    SAMPLE_RATE = 48000
    INTERVAL = 0.8
    
    THRESHOLD = 0.75
    CMD_TIMEOUT = 3
    
    REINFORCEMENT_MODE = False
    DEBUG_MODE = False
        
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cmd_dict = [row for row in reader]
    
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
        audio_raw = audio
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
            if DEBUG_MODE:
                logger.info(f"最高概率标签：{cmd_dict[pred_index]["call_sign"]}  最高概率：{max_prob:.2f}")
            if max_prob >= THRESHOLD and pred_index != 0:
                if pred_index == 14:
                     keyboardSimulator.start()
                     command_start = True
                     logger.info(f"✅  指令：")
                     ticker = time.time()
                elif command_start:
                    call_sign = cmd_dict[pred_index]["call_sign"]
                    command_sequence = cmd_dict[pred_index]["command_sequence"]
                    if not REINFORCEMENT_MODE: keyboardSimulator.read_cmd_seq(command_sequence)
                    keyboardSimulator.end()
                    command_start = False
                    logger.info(f"▶️  {call_sign}：{command_sequence}")
                    if REINFORCEMENT_MODE:
                        judgement = input(f"识别是否正确？(y/index): ")
                        if judgement == "y" or judgement == "Y": judgement = pred_index
                        else: judgement = int(judgement)
                        stratagems_name = cmd_dict[judgement]["name"]
                        audio_save_dir = os.path.join(AUDIO_SAVE_BASE, stratagems_name)
                        time_stamp = int(time.time())
                        os.makedirs(audio_save_dir, exist_ok=True)
                        audio_sav_path = f"{audio_save_dir}/{stratagems_name}-{time_stamp}-RI.{AUDIO_EXT}"
                        sf.write(audio_sav_path, audio_raw, SAMPLE_RATE)
                        audio_sav_path_copy = f"{audio_save_dir}/{stratagems_name}-{time_stamp}-RI copy.{AUDIO_EXT}"
                        sf.write(audio_sav_path_copy, audio_raw, SAMPLE_RATE)
                    
            elif command_start and time.time() - ticker > CMD_TIMEOUT:
                    keyboardSimulator.end()
                    command_start = False
                    logger.info(f"❌  指令被取消")
