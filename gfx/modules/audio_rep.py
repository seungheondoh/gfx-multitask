import os
import torch
import torchaudio
import torch.nn as nn

class TFRep(nn.Module):
    def __init__(self, 
                sample_rate= 22050,
                f_min=0,
                f_max=11050,
                n_fft = 1024,
                win_length = 1024,
                hop_length = 512,
                n_mels = 128,
                power = None,
                pad= 0,
                normalized= False,
                center= True,
                pad_mode= "reflect"
                ):
        super(TFRep, self).__init__()
        self.window = torch.hann_window(win_length)
        self.spec_fn = torchaudio.transforms.Spectrogram(
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            power = power
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels, 
            sample_rate,
            f_min,
            f_max,
            n_fft // 2 + 1)
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def melspec(self, wav):
        spec = self.spec_fn(wav)
        power_spec = spec.real.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        mel_spec = self.amplitude_to_db(mel_spec)
        return mel_spec