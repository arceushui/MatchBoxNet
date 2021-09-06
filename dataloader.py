import csv
import json
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
from vad_torch import VoiceActivityDetector
import math
import os
#os.chdir('/notebooks/small_footprint_model')


def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json, audio_conf):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json
        :param snr_list: choice of snr, e.g. snr_list = [0,-5,-10]
        :param noise_dir: musan noise dir 
        
        """
        #self.noise_dir = "noise/"
        #self.snr_list = [0, -5, -10]
        self.dataset = open(dataset_json).readlines()
        self.audio_conf = audio_conf
#         self.vad_model = vad_model
#         self.vad_utils = vad_utils
        
        print('---------------Building {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        print('json file: {:s}'.format(dataset_json))
        
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('Using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        
        self.mixup = self.audio_conf.get('mixup')
        print('Using mix-up with rate {:f}'.format(self.mixup))
        
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        print('Use dataset mean {:.3f} and std {:.3f} to normalize the input'.format(self.norm_mean, self.norm_std))
            
        ## class label
        self.Label2Indx = {
            'unknown': 0,
            'silence': 1,
            'yes':     2,
            'no':      3,
            'up':      4,
            'down':    5,
            'left':    6,
            'right':   7,
            'on':      8,
            'off':     9,
            'stop':    10,
            'go':      11}
        
    def _add_noise(self, speech_sig, vad_duration, noise_sig, snr):
        snr = 10**(snr/10.0)
        speech_power = torch.sum(speech_sig**2)/vad_duration
        noise_power = torch.sum(noise_sig**2)/noise_sig.shape[1]
        noise_update = noise_sig / torch.sqrt(snr * noise_power/speech_power)

        if speech_sig.shape[1] > noise_update.shape[1]:
            # padding
            temp_wav = torch.zeros(1, speech_sig.shape[1])
            temp_wav[0, 0:noise_update.shape[1]] = noise_update
            noise_update = temp_wav
        else:
            # cutting
            noise_update = noise_update[0, 0:speech_sig.shape[1]]

        
        return noise_update + speech_sig

    def _wav2fbank(self, filename, filename2=None):
        # mixup
#         snr = [0,-5, -10]
        seed = random.randint(1, 930)
        
        waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()
        
        """
        if random.random() < 0: 
            v = VoiceActivityDetector(waveform, sr)
            raw_detection = v.detect_speech()
            speech_labels = v.convert_windows_to_readible_labels(raw_detection)

    #             speech_labels = get_speech_ts(waveform.squeeze(), self.vad_model,
    #                                   num_steps=4)
            if len(speech_labels) == 0:
                vad_duration = 8000
            else:
                start = speech_labels[0]['speech_begin']
                end = speech_labels[0]['speech_end']
                vad_duration = end-start
                
            noise, _ = torchaudio.load(self.noise_dir + str(seed) + '.wav')
            waveform = self._add_noise(waveform, vad_duration, noise, snr=self.snr_list[random.randint(0,len(self.snr_list)-1)])
        """
                   
        # mixup
        if filename2 != None:

            waveform2, _ = torchaudio.load(filename2)
            waveform2 = waveform2 - waveform2.mean()

            if waveform.shape[1] != waveform2.shape[1]:
                if waveform.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform.shape[1]]


            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
        
    

        max_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = max_length - n_frames

        # cut and pad
        if p > 0:
            pad_top = p // 2
            pad_bottom = p // 2

            if p % 2 == 1:
                pad_bottom += 1

            m = torch.nn.ZeroPad2d((0, 0, pad_top, pad_bottom))
            fbank = m(fbank)
            
        elif p < 0:
            fbank = fbank[:, 0:max_length]
        

        if filename2 == None:
            return fbank, n_frames, 0
        else:
            return fbank, n_frames, mix_lambda

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        audio_meta = eval(self.dataset[index])
        
        ## do mix-up for this sample (controlled by the given mixup rate)
        if random.random() < self.mixup:
            # find another sample to mix, also do balance sampling
            mix_sample_idx = random.randint(0, len(self.dataset)-1)
            mix_audio_meta = eval(self.dataset[mix_sample_idx])
            
            # get the mixed fbank
            fbank, audio_length, mix_lambda = self._wav2fbank(audio_meta['audio_filepath'],
                                                mix_audio_meta['audio_filepath'])
            
            # initialize the label
            label_indices = np.zeros(len(self.Label2Indx))
            label_indices[self.Label2Indx[audio_meta['command']]] += mix_lambda
            label_indices[self.Label2Indx[mix_audio_meta['command']]] += 1.0-mix_lambda
            label_indices = torch.FloatTensor(label_indices)
        
        # skip mixup
        else:
            label_indices = np.zeros(len(self.Label2Indx))
            fbank, audio_length, mix_lambda = self._wav2fbank(audio_meta['audio_filepath'])
            label_indices[self.Label2Indx[audio_meta['command']]] = 1.0
            label_indices = torch.FloatTensor(label_indices)

        ## perform SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        if self.freqm != 0: ## apply 2 continuous freq masking 
            fbank = freqm(fbank)
        
        if self.timem != 0: ## apply 2 continuous time masking 
            fbank = timem(fbank)
        
        fbank = torch.transpose(fbank, 0, 1)

        ## normalize the input
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)


        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, audio_length, label_indices

    def __len__(self):
        return len(self.dataset)
