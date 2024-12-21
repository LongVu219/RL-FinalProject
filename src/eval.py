from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset
from utils import *


#=====================Define env and video setting================================================================================
eva_env = battle_v4.env(map_size=45, max_cycles=300)

eva_env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 35
frames = []
#===============================================================================================================


#===========================Hyper params setting=======================================================================
import base_model
import torch

device = 'cuda'

#==================================================================================================




#==================================play game=====================================================

def play_one_game(red_agent, blue_agent, policy): 
    eva_env.reset()

    red_alive, blue_alive = 81, 81
    mem = {}
    for agent in eva_env.agent_iter():
        observation, reward, termination, truncation, info = eva_env.last()
        agent_handle = agent.split("_")[0]

        if (reward > 4.5):
            if (agent_handle == 'blue'): red_alive -= 1
            else: blue_alive -= 1

        if termination or truncation:
            action = None  # this agent has died
        else:
            if agent_handle == "red":
                action = get_action(eva_env, None, agent, observation, red_agent, 'best')
            else:
                state = torch.Tensor(observation).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
                action_prob = blue_agent.policy.actor(state) 
                action = torch.argmax(action_prob, dim=-1).item() 
        
        eva_env.step(action)
    return red_alive, blue_alive

# def play_one_game(red_agent, blue_agent, policy):
#     eva_env.reset()

#     red_alive, blue_alive = 81, 81
#     mem = {}
#     for agent in eva_env.agent_iter():
#         observation, reward, termination, truncation, info = eva_env.last()
#         agent_handle = agent.split("_")[0]
#         if termination or truncation:
#             if (termination and agent not in mem):
#                 mem[agent] = 1
#                 if (agent_handle == 'red'): red_alive -= 1
#                 else: blue_alive -= 1
#             action = None  # this agent has died
#         else:
#             if agent_handle == "red":
#                 action = get_action(eva_env, None, agent, observation, red_agent, policy)
#             else:
#                 state = torch.Tensor(observation).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
#                 action = blue_agent.policy.actor(state) 
#                 action = torch.argmax(action, dim=-1).item() 
        
#         eva_env.step(action)


#     return red_alive, blue_alive

def evaluate(red_agent, blue_agent, policy, rounds, debug = False):
    red_agent = base_model.QNetwork((13, 13, 5), (21)).to(device)
    red_agent.load_state_dict(torch.load("/mnt/apple/k66/hanh/cross_q/pretrained_model/red.pt"))
    if (debug == True):
        print('==================Evaluating agent vs agent=========================')
    avg = 0
    for round in range(1, rounds + 1):
        red, blue = play_one_game(red_agent, blue_agent, policy)

        if (round % 1 == 0 and debug == True):
            print(f'Current balance of power : {(red + 1)/(blue + 1)}, {red}, {blue}')
        avg += (red + 1)/(blue + 1)

    avg /= rounds
    if (debug == True):
        print(f'Average red vs blue power projection: {avg}')
    return avg


