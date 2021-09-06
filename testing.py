#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matchbox.matchbox_model import EncDecBaseModel
import os
from time import time

import torch
from torch import nn
import argparse
import numpy as np
import random

import dataloader
from torch.utils.data import Dataset, DataLoader
#from model import KWSConvMixer
from sklearn.metrics import accuracy_score


########## Argument parser
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-data_train", type=str, default='data/train_manifest.json', help="training data json")
    parser.add_argument("-data_val", type=str, default='data/valid_manifest.json', help="validation data json")
    parser.add_argument("-data_test", type=str, default='data/test_manifest.json', help="testing data json")

    parser.add_argument('-train_batch_size', action="store_true", default=256)
    parser.add_argument('-val_batch_size', action="store_true", default=512)
    parser.add_argument('-num_workers', action="store_true", default=8)
    
    parser.add_argument('-num_classes', action="store_true", default=12)
    parser.add_argument('-saved_model', type=str, required=True)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-save_dir', type=str, default='save_models')
    parser.add_argument('-num_epochs', action="store_true", default=50)
    parser.add_argument('-save_interval', action="store_true", default=1000)
    parser.add_argument('-log_interval', action="store_true", default=200)
    parser.add_argument('-lr', action="store_true", default=0.001)
    
    args = parser.parse_args()
    return args



def test(model, device, test_loader, target_length):
    
    print('Size of testing set: ' + str(len(test_loader.dataset)))
    
    model.eval()
    correct = 0
    test_loss = 0
    start = time()
    gt_labels, pred_labels = [], []
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    with torch.no_grad():
        for batch_idx, (data, audio_len, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            output = model(data, audio_len.to(device))
            loss = criterion(output, target) 
            test_loss += loss
            pred = output.cpu().data.numpy().argmax(axis=1)
            target_lab = target.cpu().data.numpy().argmax(axis=1)
            correct += int((pred == target_lab).sum())
            
            gt_labels.append(target_lab)
            pred_labels.append(pred)
    
    end = time()
    test_loss /= len(test_loader.dataset)
    
    test_acc = accuracy_score(np.hstack(gt_labels), np.hstack(pred_labels))
    print('Test loss: {}'.format(test_loss))
    print('Test accuracy: {}'.format(test_acc))
    
    return test_acc

  
def main():
    
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Use", device)
    num_classes = {'speechcommands': 12, 'wukong': 2}
    
    """
    model = KWSConvMixer(input_size = (98, 64), 
                         num_classes=12,
                         feat_dim=64,
                         freq_filters=32,
                         time_filters=64).to(device)
    """
    model = EncDecBaseModel( num_mels= 64, final_filter = 128, num_classes=num_classes['speechcommands']).to(device)
    
    
#     model = KWSTransformer(
#               input_size = (98, 64),
#               num_classes = 12,
#               linear_dim = 128,
#               depth = 3,
#               heads = 4,
#               dropout = 0.1,
#               embedding_dropout = 0.1).to(device)

    
    #norm_stats = {'speechcommands' : [-8.522078, 4.0515576]}
    norm_stats = {'speechcommands' : [-6.52471657747699, 5.13031835310399]}
    target_length = {'speechcommands' : 128}
    noise = {'speechcommands':True}
    
    val_audio_conf = {'num_mel_bins': 64, 'target_length': target_length['speechcommands'],
                      'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'evaluation', 
                      'mean':norm_stats['speechcommands'][0], 'std':norm_stats['speechcommands'][1], 
                      'noise':False}
    
    test_loader = torch.utils.data.DataLoader(
                        dataloader.AudiosetDataset(args.data_test, audio_conf=val_audio_conf),
                        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    if not use_cuda:
        checkpoint = torch.load(args.saved_model, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(args.saved_model)
        
    model.load_state_dict(checkpoint['model'])    
    
    acc = test(model, device, test_loader, target_length)

if __name__ == '__main__':
    main()
