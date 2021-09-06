#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matchbox.matchbox_model import EncDecBaseModel
from torch.optim import optimizer
import os
from time import time

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import argparse
import numpy as np
import random
import torch.optim as optim

import dataloader
from torch.utils.data import Dataset, DataLoader
#from model import KWSConvMixer # KWSTransformer
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler

import torch_optimizer


########## Argument parser
def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    ## data dir to JSON file
    parser.add_argument("-data_train", type=str, default='data/train_manifest.json', help="training data json")
    parser.add_argument("-data_val", type=str, default='data/valid_manifest.json', help="validation data json")
    parser.add_argument("-data_test", type=str, default='data/test_manifest.json', help="testing data json")
    ## model dataset 
    parser.add_argument('-train_batch_size', action="store_true", default=256)
    parser.add_argument('-val_batch_size', action="store_true", default=512)
    parser.add_argument('-num_workers', action="store_true", default=8)
    
    parser.add_argument('-num_classes', action="store_true", default=12)
    parser.add_argument('-use_gpu', action="store_true", default=True)
    parser.add_argument('-save_dir', type=str, default='save_models')
    parser.add_argument('-num_epochs', action="store_true", default=50)
    parser.add_argument('-save_interval', action="store_true", default=1000)
    parser.add_argument('-log_interval', action="store_true", default=200)
    parser.add_argument('-lr', action="store_true", default=0.003)
    
    args = parser.parse_args()
    return args




def train(model, device, train_loader, optimizer, scaler, epoch, target_length, log_interval=100):
    ## set training mode
    model.train()  
    iteration = 0
    train_loss = 0.
    start = time()
    gt_labels, pred_labels = [], []
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    for batch_idx, (data, data_len, target) in enumerate(train_loader):
        
        correct=0
        data, target = data.to(device), target.to(device)
        
        with autocast():
            output = model(data, data_len.to(device))
            loss = criterion(output, target)
        
        train_loss += loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pred = output.cpu().data.numpy().argmax(axis=1)
        target_lab = target.cpu().data.numpy().argmax(axis=1)
        correct += int((pred == target_lab).sum())
        acc = 100. * correct / len(target)
        
        if iteration % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} Acc: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx / len(train_loader), loss.item(), acc))
    
        gt_labels.append(target_lab)
        pred_labels.append(pred)
        
        iteration += 1
    
    end = time()
    train_loss /= len(train_loader.dataset)
    print('Train loss is {} after Epoch {}'.format(train_loss, epoch))
    print('Total accuracy is {} after Epoch {}'.format(
        accuracy_score(np.hstack(gt_labels), np.hstack(pred_labels)), epoch))
    print('Time taken for Epoch {} is {}s'.format(epoch, end - start))

    
def test(model, device, test_loader, ep, target_length):
    
    print('Size of validation set: ' + str(len(test_loader.dataset)))
    model.eval()
    correct = 0
    test_loss = 0
    start = time()
    gt_labels, pred_labels = [], []
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    with torch.no_grad():
        for batch_idx, (data, data_len, target) in enumerate(test_loader):

            data, target = data.to(device), target.to(device)

            output = model(data, data_len.to(device))
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
    print('Valid loss is {} after Epoch {}'.format(test_loss, ep))
    print('Valid accuracy is {} after Epoch {}'.format(test_acc, ep))
    
    
    return test_acc



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    
    args = parse_args()
    num_classes = {'speechcommands': 12, 'wukong': 2}
    setup_seed(2345)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Training using: ", device)
    
    """
    model = KWSConvMixer(input_size = (98, 64), 
                         num_classes=12,
                         feat_dim=64,
                         freq_filters=32,
                         time_filters=64).to(device)
    """

    model = EncDecBaseModel( num_mels= 64, final_filter = 128, num_classes=num_classes['speechcommands']).to(device)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    
    #norm_stats = {'speechcommands' : [-8.522078, 4.0515576]}
    norm_stats = {'speechcommands' : [-6.52471657747699, 5.13031835310399]}
    target_length = {'speechcommands' : 128}
    noise = {'speechcommands':False}

    audio_conf = {'num_mel_bins': 64, 'target_length': target_length['speechcommands'],
                  'freqm': 25, 'timem': 25, 'mixup': 0, 'mode':'train',
                  'mean':norm_stats['speechcommands'][0], 'std':norm_stats['speechcommands'][1],
                  'noise':noise['speechcommands']}
    
    val_audio_conf = {'num_mel_bins': 64, 'target_length': target_length['speechcommands'],
                      'freqm': 0, 'timem': 0, 'mixup': 0, 'mode':'evaluation', 
                      'mean':norm_stats['speechcommands'][0], 'std':norm_stats['speechcommands'][1], 
                      'noise':False}

    train_loader = torch.utils.data.DataLoader(
                        dataloader.AudiosetDataset(args.data_train, audio_conf=audio_conf),
                        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
                        dataloader.AudiosetDataset(args.data_val, audio_conf=val_audio_conf),
                        batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    
    ## training optimizer :: adamW
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
    optimizer = torch_optimizer.NovoGrad(model.parameters(), lr=0.05, betas=(0.95, 0.5), weight_decay=0.001)
 
    ## make decay in learning rate
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5, 26)), gamma=0.85)

    scaler = GradScaler()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    
    print('Starting model training ...')
    for epoch in range(args.num_epochs):
        print('\n=============start training=============')
        train(model, device, train_loader, optimizer, scaler, epoch, target_length, args.log_interval)
        
        print('\n============start validation============')
        acc = test(model, device, val_loader, epoch, target_length)  
        
        scheduler.step()
        
        ## saving model
        model_save_path = os.path.join(args.save_dir, 'check_point_'+str(epoch)+'_'+str(acc))
        state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state_dict, model_save_path)
    
if __name__ == '__main__':
    main()
