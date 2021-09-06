## compute mean and std
from tqdm import tqdm
import torchaudio
import torch
import numpy as np

mean = []
std = []
min_length = 100
data = open('data/train_manifest.json').readlines()
for file in tqdm(data):
    waveform, sr = torchaudio.load(eval(file)['audio_filepath'])
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True,
                                              sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=64,
                                              dither=0.0, frame_shift=10)
    
    max_length = 128
    n_frames = fbank.shape[0]
    
    min_length = min(min_length, n_frames)

    p = max_length - n_frames

    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    
    mean.append(fbank.mean().item())
    std.append(fbank.std().item())
print(np.mean(mean))
print(np.mean(std))
