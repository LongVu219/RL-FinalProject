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
from base_model import *
import torch
from train_net import *
from evaluate_fight import *

def get_agent(agent_path, agent_type):
    get_env = env = battle_v4.env(map_size=45, max_cycles=300)
    get_env.reset()
    agent = agent_type(
        get_env.observation_space("red_0").shape, get_env.action_space("red_0").n
    ).to(device)
    if (agent_path is not None):
        agent.load_state_dict(
            torch.load(agent_path, weights_only=True, map_location=device)
        )
    return agent

def train_agent(red_agent, blue_agent, save_path, use_red = True, train_red=False, train_blue=True, episodes = 30, selfplay = False):
    env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
    dead_penalty=-0.5, attack_penalty=-0.1, attack_opponent_reward=0.5,
    max_cycles=300, extra_features=False, render_mode = "rgb_array")
    num_agent = 162
    env.reset()

    device = 'cuda'

    lr = 0.001
    loss_function = nn.MSELoss()

    #config red
    red_optimizer = optim.Adam(red_agent.parameters(), lr=lr)

    #config blue
    blue_optimizer = optim.Adam(blue_agent.parameters(), lr=lr)
    
    best_score = 100
    for episode in range (1, episodes + 1):

        print(f'Episode number {episode} running...................................')
        
        #gathering training data
        env.reset()
        X, y = [], []

        #because the reward is last reward, not te current reward so we have to trace backward
        buffer: dict[str, list[tuple]] = {}
        for id, agent in enumerate(env.agent_iter()):
            observation, reward, termination, truncation, info = env.last()
            
            agent_handle = agent.split("_")[0]
            if termination or truncation:
                action = None  # this agent has died
            else:
                if agent_handle == "red":
                    if (train_red == False):
                        action = get_action(env, episode, agent, observation, red_agent, 'best')
                    else:
                        action = get_action(env, episode, agent, observation, red_agent, 'epsilon')
                else:
                    if (train_blue == False):
                        action = get_action(env, episode, agent, observation, blue_agent, 'best')
                    else:
                        action = get_action(env, episode, agent, observation, blue_agent, 'epsilon')
            
            if (agent not in buffer):
                buffer[agent] = []
            buffer[agent].append((agent, observation, action, reward, termination, truncation, info))
            env.step(action)

        for agent in buffer.keys():
            state_array = buffer[agent]
            agent_handle = agent.split("_")[0]
            
            if (agent_handle == 'red' and use_red == False): continue

            for i in range(0, len(state_array)):
                agent, observation, action, prv_reward, termination, truncation, info = state_array[i]
                _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i]
                if (i < len(state_array) - 1):
                    _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i + 1]

                observation = convert_obs(observation)
                nxt_observation = convert_obs(nxt_observation)

                with torch.no_grad():
                    next_max = blue_agent(nxt_observation).squeeze(dim = 0).max()
                    tmp = blue_agent(observation).squeeze(dim = 0)
                    if (i == len(state_array)):
                        next_max = 0
                    tmp[action] = reward + next_max
                    X.append(observation.squeeze(dim = 0))
                    y.append(tmp)

        X_tensor = torch.stack(X)
        y_tensor = torch.stack(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

        #Train model with given data
        if (train_red):
            train_model(50, dataloader, red_agent, red_optimizer, lr, loss_function)

        if (train_blue):
            train_model(50, dataloader, blue_agent, blue_optimizer, lr, loss_function)

        if (episode % 1 == 0):
            avg = evaluate(red_agent=red_agent, blue_agent=blue_agent, rounds=3, debug = True)
            if (avg < best_score):
                torch.save(blue_agent.state_dict(), save_path)
                print('Agent saved !')
                print()
                best_score = avg
            
            torch.save(blue_agent.state_dict(), 'model/newest.pth')

    print(best_score)
    env.close()

def selfplay(red_agent, blue_agent, save_path, episodes = 10):
    best_score = 100
    for episode in range (1, episodes + 1):
        print(f'/\/\/\/\/\/\/\/\/ selfplay round number {episode} /\/\/\/\/\/\/\/\/')
        train_agent(red_agent, blue_agent, save_path, 
                    train_red=True, train_blue=False, episodes=1, selfplay=True)
        train_agent(red_agent, blue_agent, save_path, 
                    train_red=False, train_blue=True, episodes=1, selfplay=True)

        eva_agent = get_agent('model/red_final.pt', QNetwork_final)
        avg = evaluate(red_agent=eva_agent, blue_agent=blue_agent, rounds=3, debug = True)
        if (avg < best_score):
            torch.save(blue_agent.state_dict(), save_path)
            print('Agent saved !')
            print()
            best_score = avg
