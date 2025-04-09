from devtoolkit.Log4P import Log4P
from nn.dataset.Preprocessing import Preprocessing

import sounddevice as sd
import soundfile as sf
import noisereduce as nr
import time
import os

if __name__ == "__main__":
    
    SAMPLE_RATE = 48000
    INTERVAL = 3
    
    SAVE_BASE = "src/resources/dat"
    STRATAGEMS_NAME = "tmp"
    AUDIO_EXT = "wav"
    
    logger = Log4P(enable_level = True,
                   enable_timestamp = True,
                   enable_source = True,
                   enable_log_file = False,
                   source = "main",)
    logger.info("🎙️ 开始监听麦克风...")
    audio_save_dir = os.path.join(SAVE_BASE, STRATAGEMS_NAME)
    os.makedirs(audio_save_dir, exist_ok=True)
    for epoch in range(1):
        start = time.time()
        logger.info(f"{epoch} start")
        audio = sd.rec(int(SAMPLE_RATE * INTERVAL), samplerate = SAMPLE_RATE, channels = 2)
        sd.wait()
        # audio = nr.reduce_noise(y=audio.flatten(), 
        #                         sr = SAMPLE_RATE, 
        #                         y_noise = audio[:int(SAMPLE_RATE * 0.5)])
        time_stamp = int(time.time())
        audio_sav_path = f"{audio_save_dir}/{STRATAGEMS_NAME}-{time_stamp}.{AUDIO_EXT}"
        sf.write(audio_sav_path, audio, SAMPLE_RATE)
        print(time.time() - start)
        