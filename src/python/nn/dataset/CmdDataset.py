from nn.dataset.Preprocessing import Preprocessing

from torch.utils.data import Dataset
import torchaudio
import csv

class CmdDataset(Dataset):
    
    DICT_PATH = "src/resources/cmd-dict.csv"
    
    def __init__(self, 
                 dataframe,
                 sample_rate,
                 length):
        self.dataframe = dataframe
        self.sample_rate = sample_rate
        self.length = length
        self.label_mapper = {}
        
        # 前有自动创建标签-索引映射的预感
        with open(self.DICT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.label_mapper[row["name"]] = int(row["index"])

        
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.dataframe.loc[idx, "sample_path"])
        
        # 献上对音频数据的预处理
        audio = Preprocessing.resample(audio = audio, 
                                       orig_sample_rate = sr, 
                                       new_sample_rate=self.sample_rate)
        
        # audio_np = audio.numpy()[0]
        # audio_np = Preprocessing.AUDIO_AUGMENT(samples=audio_np, 
        #                                        sample_rate=self.sample_rate)
        
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