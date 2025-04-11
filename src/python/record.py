from devtoolkit.Log4P import Log4P

import sounddevice as sd
import soundfile as sf
import random
import time
import os
import csv

if __name__ == "__main__":
    
    SAMPLE_RATE = 48000
    INTERVAL = 1
    
    SAVE_BASE = "src/resources/dat"
    AUDIO_EXT = "wav"
    DICT_PATH = "src/resources/cmd-dict.csv"
    
    logger = Log4P(enable_level = True,
                   enable_timestamp = True,
                   enable_source = True,
                   enable_log_file = False,
                   source = "main",)
    
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cmd_dict = [row for row in reader]
        
    logger.info("🎙️ 开始监听麦克风...")
    epoch = 0
    logger.info("倒计时：3秒")
    time.sleep(1)
    logger.info("倒计时：2秒")
    time.sleep(1)
    logger.info("倒计时：1秒")
    time.sleep(1)
    while True:
        epoch += 1
        stratagems_index = random.randint(0, 25)                  # 随机战备
        # stratagems_index = 14                                       # 指定战备索引
        stratagems_name = cmd_dict[stratagems_index]["name"]
        stratagems_call_sign = cmd_dict[stratagems_index]["call_sign"]
        audio_save_dir = os.path.join(SAVE_BASE, stratagems_name)
        os.makedirs(audio_save_dir, exist_ok=True)
        
        logger.info(f"Epoch {epoch} start: {stratagems_call_sign}")
        start = time.time()
        audio = sd.rec(int(SAMPLE_RATE * INTERVAL), samplerate = SAMPLE_RATE, channels = 2)
        sd.wait()
        time_stamp = int(time.time())
        audio_sav_path = f"{audio_save_dir}/{stratagems_name}-{time_stamp}.{AUDIO_EXT}"
        logger.info(f"样本长度:{time.time() - start:.2f} 秒")
        sf.write(audio_sav_path, audio, SAMPLE_RATE)
        
        # 临时的数据增强手段，记得注释掉！(不过似乎效果挺好？)
        audio_sav_path_copy = f"{audio_save_dir}/{stratagems_name}-{time_stamp} copy.{AUDIO_EXT}"
        sf.write(audio_sav_path_copy, audio, SAMPLE_RATE)