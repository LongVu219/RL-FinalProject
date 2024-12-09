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
eva_env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-0.2, attack_penalty=-0.1, attack_opponent_reward=3.5,
max_cycles=200, extra_features=False, render_mode = "rgb_array")
eva_env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 35
frames = []
#===============================================================================================================


#===========================Hyper params setting=======================================================================
import base_model
import resnet 
import torch

device = 'cuda'

#==================================================================================================




#==================================play game=====================================================

def play_one_game(red_agent, blue_agent):
    eva_env.reset()

    red_alive, blue_alive = 81, 81
    mem = {}
    for agent in eva_env.agent_iter():
        observation, reward, termination, truncation, info = eva_env.last()
        agent_handle = agent.split("_")[0]
        if termination or truncation:
            if (termination and agent not in mem):
                mem[agent] = 1
                if (agent_handle == 'red'): red_alive -= 1
                else: blue_alive -= 1
            action = None  # this agent has died
        else:
            if agent_handle == "red":
                action = get_action(eva_env, None, agent, observation, red_agent, 'random')
            else:
                action = get_action(eva_env, None, agent, observation, blue_agent, 'best')
        
        eva_env.step(action)


    return red_alive, blue_alive

def evaluate(red_agent, blue_agent, rounds, debug = False):
    if (debug == True):
        print('==================Evaluating agent vs agent=========================')
    avg = 0
    for round in range(1, rounds + 1):
        red, blue = play_one_game(red_agent, blue_agent)

        if (round % 1 == 0 and debug == True):
            print(f'Current balance of power : {(red + 1)/(blue + 1)}, {red}, {blue}')
        avg += (red + 1)/(blue + 1)

    avg /= rounds
    if (debug == True):
        print(f'Average red vs blue power projection: {avg}')
    return avg