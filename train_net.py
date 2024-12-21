from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset
import time
from utils import *

device = 'cuda'

def train_1_epoch(num, dataloader, model, optimizer, lr, loss_fn):

    debug = False
    if (num%50 == 0): debug = True
    total_loss = 0
    for id, (X, y) in enumerate(dataloader):
        model.train()
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss / len(dataloader)
    if (debug == True):
        print(f'Epoch number {num} loss : {total_loss}')
    
    return total_loss

def train_model(num_epoch, dataloader, model, optimzer, lr, loss_fn):
    current_loss = 100
    total_epoch = 0
    while(current_loss > 0.005 and total_epoch < 500):
        for epoch in range(1, num_epoch + 1):
            current_loss = train_1_epoch(epoch, dataloader, model, optimzer, lr, loss_fn)
        
        total_epoch += num_epoch