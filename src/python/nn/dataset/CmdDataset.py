from nn.dataset.Preprocessing import Preprocessing

from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

class CmdDataset(Dataset):
    
    label_mapper = {
            "silent": 0,            # 白噪音
            "rl-77": 1,             # 空爆
            "a-mls-4x": 2,          # 火箭炮塔
            "a-mg-43": 3,           # 机枪炮塔
            "a-ac-8": 4,            # 加农炮塔
            "gr-8": 5,              # 无后座
            "a-m-23": 6,            # 电磁迫击炮
            "sos": 7,               # SOS
            "reinforce": 8,         # 增援
            "resupply": 9,          # 补给
            "hellbomb": 10,         # 地狱火
            "orbit-laser": 11,      # 轨道激光
            "orbit-nap": 12,        # 轨道汽油弹
            "500kg": 13,            # 500千克
            "command": 14,          # 指令
        }
    
    def __init__(self, 
                 dataframe,
                 sample_rate,
                 length):
        self.dataframe = dataframe
        self.sample_rate = sample_rate
        self.length = length
        
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.dataframe.loc[idx, "sample_path"])
        
        # 献上对音频数据的预处理
        audio = Preprocessing.resample(audio = audio, 
                                       orig_sample_rate = sr, 
                                       new_sample_rate=self.sample_rate)
        audio = Preprocessing.isometricalization(audio = audio,
                                                 sample_rate = self.sample_rate,
                                                 length = self.length)
        audio = Preprocessing.mono(audio = audio)
        audio = Preprocessing.mel_spectrogram(audio = audio,
                                              sample_rate = self.sample_rate)
        
        # 最后是对标签的映射
        label_id = self.label_mapper[self.dataframe.loc[idx, "label"]]
        return audio, label_id
    
    def __len__(self):
        return len(self.dataframe["sample_path"])