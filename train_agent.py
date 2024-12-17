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
from train_net import *
from train_episode import *


def debug(var):
    print(var)
    sys.exit()

start_time = time.time()

red_path = 'model/red.pt'
blue_path = None
first_save_path = 'model/agent_base.pth'
train_agent(red_path, blue_path, save_path=first_save_path, 
            train_red = False, train_blue = True, episodes = 20)

print('//' * 60)

'''
red_path = first_save_path
blue_path = first_save_path
sec_save_path = 'model/agent_selfplay.pth'
train_agent(red_path, blue_path, save_path = sec_save_path,
            train_red = True, train_blue = True, episodes = 100)
'''
end_time = time.time()

print(f'Total running time : {(end_time - start_time)/3600}hrs')

