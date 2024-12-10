from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset
from utils import *
from ppo import PPO 


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
import torch

ppo = PPO(
    observation_shape = eva_env.observation_space("blue_0").shape, 
    action_shape = eva_env.action_space("blue_0").n, 
    episodes = 300,
    batch_size = 128, 
    lr_actor = 1e-5, 
    lr_critic = 1e-5, 
    epsilon_clip = 0.2, 
    epochs = 50,
    device = "cuda"
)

ppo.load('/home/ubuntu/school/RL/RL-FinalProject/model/blue.pt') 

red_agent = base_model.QNetwork(
    observation_shape = eva_env.observation_space("blue_0").shape, 
    action_shape = eva_env.action_space("blue_0").n,
).to("cuda")

red_agent.load_state_dict("/home/ubuntu/school/RL/RL-FinalProject/model/red.pt")

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
                state = torch.Tensor(observation).float().unsqueeze(0).permute(0, 3, 1, 2).to("device")
                action = blue_agent.policy_old.actor(state) 
                action = torch.argmax(action, dim=-1).item() 
        
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

evaluate(red_agent=red_agent, blue_agent=ppo, rounds=100, debug=True)
