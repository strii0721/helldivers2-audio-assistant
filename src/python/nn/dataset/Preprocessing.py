from scipy.signal import butter, lfilter
import torchaudio
import torch
import torchaudio.transforms as T

class Preprocessing:
    
    # 前有重采样的预感
    @staticmethod
    def resample(audio, orig_sample_rate, new_sample_rate):
        audio = torchaudio.functional.resample(audio, 
                                               orig_freq = orig_sample_rate, 
                                               new_freq = new_sample_rate)
        return audio
    
    # 截断与填充万岁！也就是说，将样本处理为等长片段
    @staticmethod
    def isometricalization(audio, sample_rate, length):
        desired_num_samples = int(sample_rate * length)
        if audio.shape[1] > desired_num_samples:
            audio = audio[:, :desired_num_samples]
        elif audio.shape[1] < desired_num_samples:
            padding = desired_num_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, padding))
        return audio
    
    # 以将音频转化为单声道为目标吧
    @staticmethod
    def mono(audio):
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio
    
    # 居然是梅尔滤波器，敬请见证
    @staticmethod
    def mel_spectrogram(audio, sample_rate):
        transform = T.MelSpectrogram(sample_rate = sample_rate, 
                                     n_mels=64, 
                                     n_fft=1024, 
                                     hop_length=64)
        audio = transform(audio)
        return audio
    
    @staticmethod
    def bandpass_filter(audio, sample_rate, high_cut, low_cut, order = 6):
        nyquist = 0.5 * sample_rate
        low = low_cut / nyquist
        high = high_cut / nyquist

        b, a = butter(order, [low, high], btype='band')  # 带通滤波器
        filtered_audio = lfilter(b, a, audio)
        return filtered_audio