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
env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-5, attack_penalty=-0.1, attack_opponent_reward=0.3,
max_cycles=30, extra_features=False, render_mode = "rgb_array")
num_agent = 160
env.reset()
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
#base model
blue_agent = base_model.QNetwork(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
).to(device)
blue_agent.load_state_dict(
    torch.load("model/red.pt", weights_only=True, map_location=device)
)
blue_policy = "random"

#current training model
red_agent = resnet.QNetwork(
    env.observation_space("red_0").shape, env.action_space("red_0").n
).to(device)

#load sth if needed

red_policy = "best"
#==================================================================================================




#==================================play game=====================================================

def play_one_game(red_agent, blue_agent, env):
    print('PLaying another game......................')
    env.reset()

    red_alive, blue_alive = 0, 0

    cnt = 0
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        agent_handle = agent.split("_")[0]
        if termination or truncation:
            if (truncation):
                if (termination == False):
                    if (agent_handle == 'red'): red_alive += 1
                    else: blue_alive += 1 
            action = None  # this agent has died
        else:
            
            if agent_handle == "red":
                action = get_action(env, agent, observation, red_agent, 'best')
            else:
                action = get_action(env, agent, observation, blue_agent, 'random')
        
        env.step(action)
        cnt += 1
        if (cnt % 500 == 0):
            print(f'Playing step {cnt}...')

    return red_alive, blue_alive

def evaluate(red_agent, blue_agent, env, rounds):
    red_win = 0
    draw = 0
    blue_win = 0
    for round in range(1, rounds + 1):
        red, blue = play_one_game(red_agent, blue_agent, env)
        if (red > blue): red_win += 1
        if (red == blue): draw += 1
        if (red < blue): blue_win += 1

        if (round % 1 == 0):
            print(f'Current situation : {red} vs {blue}')

    print('Evaluate result : -----------------')
    print(f'Red winrate : {red_win/rounds}')
    print(f'Draw rate : {draw/rounds}')
    print(f'Blue winrate : {blue_win/rounds}')


evaluate(blue_agent, red_agent, env, 50)

'''
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"pretrained-retest.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording pretrained agents")

    env.close()
'''