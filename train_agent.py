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
from base_model import *
from evaluate_fight import *

def debug(var):
    print(var, type(var))
    sys.exit()

device = 'cuda'


red_agent = get_agent('model/red.pt', QNetwork)
blue_agent = get_agent(None, Resnet)
save_path = 'model/agent_kamikaze.pth'
train_agent(red_agent, blue_agent, save_path = save_path,
            use_red = True, episodes = 30)
print('//' * 60)

'''
red_agent = get_agent('model/red_final.pt', QNetwork_final)
blue_agent = get_agent('model/agent_base_final.pth', Resnet)

save_path = 'model/agent_blue_red_redfinal.pth'
train_agent(red_agent, blue_agent, save_path = save_path,
            use_red = False, episodes = 20)
'''
best_blue = get_agent(save_path, Resnet)
print('Eval vs random')
evaluate(get_agent('model/red.pt', QNetwork), best_blue, rounds=30, red_policy='random', debug=True)
print()

print('Eval vs red.pt')
evaluate(get_agent('model/red.pt', QNetwork), best_blue, rounds=30, debug=True)
print()

print('Eval vs red_final.pt')
evaluate(get_agent('model/red_final.pt', QNetwork_final), best_blue, rounds=30, debug=True)


print('//' * 60)

'''
red_path = sec_save_path
blue_path = sec_save_path
last_save_path = 'model/agent_stage3.pth'
train_agent(red_path, blue_path, save_path = last_save_path,
            train_red = True, train_blue = True, episodes = 20)
'''


