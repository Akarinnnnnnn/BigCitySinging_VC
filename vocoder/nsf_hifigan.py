import os
import numpy as np
import torch
import librosa
import pyworld as pw
import soundfile as sf

from vocoder.modules.models import load_model
from mel_processing import mel_spectrogram_torch
import config as cfg
def load_wav_to_torch(full_path, target_sr=None, return_empty_on_exception=False):
    sampling_rate = None
    try:
        data, sampling_rate = sf.read(full_path, always_2d=True)# than soundfile.
    except Exception as ex:
        print(f"'{full_path}' failed to load.\nException:")
        print(ex)
        if return_empty_on_exception:
            return [], sampling_rate or target_sr or 48000
        else:
            raise Exception(ex)
    
    if len(data.shape) > 1:
        data = data[:, 0]
        assert len(data) > 2# check duration of audio file is > 2 samples (because otherwise the slice operation was on the wrong dimension)
    
    if np.issubdtype(data.dtype, np.integer): # if audio data is type int
        max_mag = -np.iinfo(data.dtype).min # maximum magnitude = min possible value of intXX
    else: # if audio data is type fp32
        max_mag = max(np.amax(data), -np.amin(data))
        max_mag = (2**31)+1 if max_mag > (2**15) else ((2**15)+1 if max_mag > 1.01 else 1.0) # data should be either 16-bit INT, 32-bit INT or [-1 to 1] float32
    
    data = torch.FloatTensor(data.astype(np.float32))/max_mag
    
    if (torch.isinf(data) | torch.isnan(data)).any() and return_empty_on_exception:# resample will crash with inf/NaN inputs. return_empty_on_exception will return empty arr instead of except
        return [], sampling_rate or target_sr or 48000
    if target_sr is not None and sampling_rate != target_sr:
        data = torch.from_numpy(librosa.core.resample(data.numpy(), orig_sr=sampling_rate, target_sr=target_sr))
        sampling_rate = target_sr
    
    return data, sampling_rate



def get_f0(wav):
    # wav, sr = sf.read(wavpath)
    # assert sr == cfg.sample_rate, "wav sample rate != config sample rate"
    f0, t = pw.harvest(
        wav.astype(np.double),
        cfg.sample_rate,
        f0_floor=50.0, f0_ceil=1100.0, 
        frame_period=1000 * cfg.hop_size / cfg.sample_rate,
    )
    return f0

def get_mel(wav):
    mel_spec = mel_spectrogram_torch(
        wav,
        n_fft=cfg.fft_size,
        num_mels=cfg.mel_bins,
        sampling_rate=cfg.sample_rate,
        hop_size=cfg.hop_size,
        win_size=cfg.win_size,
        fmin=cfg.fmin,
        fmax=cfg.fmax
    )
    return mel_spec

class NsfHifiGAN(object):
    def __init__(self, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        model_path = cfg.vocoder_model_path
        assert os.path.exists(model_path), 'HifiGAN model file is not found!'
        print('| Load HifiGAN: ', model_path)
        self.model, self.h = load_model(model_path, device=self.device)

    def spec2wav_torch(self, mel, f0):  # mel: [B, T, bins]
        if self.h.sampling_rate != cfg.sample_rate:
            print('Mismatch parameters: cfg.sample_rate=', cfg.sample_rate, '!=',
                  self.h.sampling_rate, '(vocoder)')
        if self.h.num_mels != cfg.mel_bins:
            print('Mismatch parameters: cfg.audio_num_mel_bins=', cfg.mel_bins, '!=',
                  self.h.num_mels, '(vocoder)')
        if self.h.n_fft != cfg.fft_size:
            print('Mismatch parameters: cfg.fft_size=', cfg.fft_size, '!=', self.h.n_fft, '(vocoder)')
        if self.h.win_size != cfg.win_size:
            print('Mismatch parameters: cfg.win_size=', cfg.win_size, '!=', self.h.win_size,
                  '(vocoder)')
        if self.h.hop_size != cfg.hop_size:
            print('Mismatch parameters: cfg.hop_size=', cfg.hop_size, '!=', self.h.hop_size,
                  '(vocoder)')
        if self.h.fmin != cfg.fmin:
            print('Mismatch parameters: cfg.fmin=', cfg.fmin, '!=', self.h.fmin, '(vocoder)')
        if self.h.fmax != cfg.fmax:
            print('Mismatch parameters: cfg.fmax=', cfg.fmax, '!=', self.h.fmax, '(vocoder)')
        with torch.no_grad():
            c = mel.transpose(2, 1)  # [B, T, bins]
            # log10 to log mel
            # c = 2.30259 * c
            # f0 = kwargs.get('f0')  # [B, T]
            y = self.model(c, f0).view(-1)

        return y#.cpu().numpy()


    # @staticmethod
    # def wav2spec(inp_path, keyshift=0, speed=1, device=None):
    #     if device is None:
    #         device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     sampling_rate = cfg.sample_rate
    #     num_mels = cfg.mel_bins
    #     n_fft = cfg.fft_size
    #     win_size = cfg.win_size
    #     hop_size = cfg.hop_size
    #     fmin = cfg.fmin
    #     fmax = cfg.fmax
    #     stft = STFT(sampling_rate, num_mels, n_fft, win_size, hop_size, fmin, fmax)
    #     with torch.no_grad():
    #         wav_torch, _ = load_wav_to_torch(inp_path, target_sr=stft.target_sr)
    #         mel_torch = stft.get_mel(wav_torch.unsqueeze(0).to(device), keyshift=keyshift, speed=speed).squeeze(0).T
    #         # log mel to log10 mel
    #         mel_torch = 0.434294 * mel_torch
    #         return wav_torch.cpu().numpy(), mel_torch.cpu()
        


        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hifigan = NsfHifiGAN(device=device)
    print("load success")
    # wav, spec = hifigan.wav2spec('Nostalgia_0_乐正绫_1.wav')
    # print(spec)
    wav_torch, _ = load_wav_to_torch('Nostalgia_0_乐正绫_1.wav', target_sr=cfg.sample_rate)
    spec = get_mel(wav_torch.unsqueeze(0)).transpose(1,2).squeeze(0)
    # print(0.434294 *mel)
    print(spec.shape)
    f0 = get_f0(wav_torch.cpu().numpy())
    min_len = min(spec.shape[0], f0.shape[0])
    spec = spec[:min_len,:]
    f0 = f0[:min_len]
    print(f0.shape)
    f0 = torch.Tensor(f0).unsqueeze(0).to(device)
    wav = hifigan.spec2wav_torch(spec.unsqueeze(0).to(device), f0)
    print(wav.shape)
    sf.write('hifigan_output.wav', wav.cpu().numpy(), cfg.sample_rate)
    # print(1)

