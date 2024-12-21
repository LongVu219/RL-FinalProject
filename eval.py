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

#eval best model, run python eval.py

save_path = 'model/agent_kamikaze.pth'
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



