from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import torch
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda'

def numpy_cpu(array):
    return array.detach().cpu().numpy()

def convert_obs(observation):
    return (
            torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
        ).to(device)

def get_action(envir, episode, agent, observation, network : nn.Module, policy : str):
    if (policy == 'random'):
        return envir.action_space(agent).sample()

    #define eps-greedy params here
    if (policy == 'epsilon'):
        eps = max(0.1, 0.9 - episode/300)
        rd = random.random()
        if (rd < eps): 
            return envir.action_space(agent).sample()
        
        observation = convert_obs(observation)
        with torch.no_grad():
            q_values = network(observation)
        return torch.argmax(q_values, dim=1).detach().cpu().numpy()[0]

    if (policy == 'best'):
        observation = convert_obs(observation)
        with torch.no_grad():
            q_values = network(observation)
        return torch.argmax(q_values, dim=1).detach().cpu().numpy()[0]