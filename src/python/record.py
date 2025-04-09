from devtoolkit.Log4P import Log4P

import sounddevice as sd
import soundfile as sf
import time
import os

if __name__ == "__main__":
    
    SAMPLE_RATE = 48000
    INTERVAL = 0.8
    
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
    for epoch in range(64):
        start = time.time()
        logger.info(f"{epoch} start")
        audio = sd.rec(int(SAMPLE_RATE * INTERVAL), samplerate = SAMPLE_RATE, channels = 2)
        sd.wait()
        time_stamp = int(time.time())
        audio_sav_path = f"{audio_save_dir}/{STRATAGEMS_NAME}-{time_stamp}.{AUDIO_EXT}"
        sf.write(audio_sav_path, audio, SAMPLE_RATE)
        print(time.time() - start)
        