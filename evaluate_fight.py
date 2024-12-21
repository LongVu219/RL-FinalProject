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
eva_env = battle_v4.env(map_size=45, max_cycles=300, render_mode = "rgb_array")
eva_env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 35
frames = []
#===============================================================================================================


#==================================play game=====================================================

def play_one_game(red_agent, blue_agent, red_policy = 'best'):
    eva_env.reset()

    red_alive, blue_alive = 81, 81
    mem = {}
    mem['red'] = 0
    mem['blue'] = 0
    for agent in eva_env.agent_iter():
        observation, reward, termination, truncation, info = eva_env.last()
        agent_handle = agent.split("_")[0]

        mem[agent_handle] += reward
        if (reward >= 4.5):
            if (agent_handle == 'blue'): red_alive -= 1
            else: blue_alive -= 1

        if termination or truncation:
            action = None  # this agent has died
        else:
            if agent_handle == "red":
                action = get_action(eva_env, None, agent, observation, red_agent, red_policy)
            else:
                action = get_action(eva_env, None, agent, observation, blue_agent, 'best')
        
        eva_env.step(action)


    return red_alive, blue_alive, mem['red'], mem['blue']

def evaluate(red_agent, blue_agent, rounds, red_policy='best', debug = False):
    if (debug == True):
        print('==================Evaluating agent vs agent=========================')
    avg = 0
    win, lose, draw = 0, 0, 0
    for round in range(1, rounds + 1):
        red, blue, r_reward, b_reward = play_one_game(red_agent, blue_agent, red_policy)

        if (round % 1 == 0 and debug == True):
            print(f'Current balance of power : {(red + 1)/(blue + 1)}, {red}, {blue}, {r_reward}, {b_reward}')
        avg += (red + 1)/(blue + 1)

        if (red > blue + 5): lose += 1
        elif (red + 5 <= blue): win += 1
        else: draw += 1

    avg /= rounds
    if (debug == True):
        print(f'Average red vs blue power projection: {avg}')
        print(f'Win rate : {win/rounds} | Lose rate : {lose/rounds} | Draw rate : {draw/rounds}')
    return avg

