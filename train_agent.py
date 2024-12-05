from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset



env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-5, attack_penalty=-0.1, attack_opponent_reward=0.3,
max_cycles=500, extra_features=False, render_mode = "rgb_array")
num_agent = 160
env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 35
frames = []

#===========================Hyper params setting=======================================================================
# pretrained policies
frames = []
env.reset()
import base_model
import resnet 
import torch

#training hyper params
device = 'cuda'
episodes = 100


#base model
base_q_network = base_model.QNetwork(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
).to(device)
base_q_network.load_state_dict(
    torch.load("model/red.pt", weights_only=True, map_location=device)
)

#current training model
better_agent = resnet.QNetwork(
    env.observation_space("red_0").shape, env.action_space("red_0").n
).to(device)
lr = 0.001
optimizer = optim.Adam(better_agent.parameters(), lr=lr)
loss_function = nn.MSELoss()
#==================================================================================================





#============Ultiliy function==============================================================================

from utils import *

#==========================================================================================================


#===================Training Neural Net - put it to another file soon==============================================
def train_1_epoch(num, dataloader, model, optimizer, lr, loss_fn):
    print(f'Epoch number {num} Training........................................')
    total_loss = 0
    for id, (X, y) in enumerate(dataloader):
        model.train()
        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss / len(dataloader)
        if (id % 5 == 0):
            print(f'Batch number {id} Loss : {loss}')
    print(f'Epoch loss : {total_loss}')

def train_model(num_epoch, dataloader, model, optimzer, lr, loss_fn):
    for epoch in range(1, num_epoch + 1):
        train_1_epoch(epoch, dataloader, model, optimzer, lr, loss_fn)
#===============================================================================================================

for episode in range (1, episodes + 1):
    #gathering training data
    env.reset()
    X, y = [], []
    buffer = []

    #because the reward is last reward, not te current reward so we have to trace backward
    for id, agent in enumerate(env.agent_iter()):
        observation, reward, termination, truncation, info = env.last()
        
        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                action = get_action(env, agent, observation, better_agent, 'epsilon')
            else:
                action = get_action(env, agent, observation, base_q_network, 'best')
        
        if (id//200 == 1):
            print(f'Finish iter number {id}')
            break
        

        buffer.append((agent, observation, action, reward, termination, truncation, info))
        env.step(action)

    #print(len(buffer))
    for (i, content) in enumerate(buffer):
        agent, observation, action, prv_reward, termination, truncation, info = content
        
        if (termination or truncation or i + num_agent >= len(buffer)): 
            continue
        else:
            _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = buffer[i + num_agent]

            observation = convert_obs(observation)
            nxt_observation = convert_obs(nxt_observation)

            #print(nxt_observation.shape, i)
            #input size (batch_size, 5, 13, 13)
            with torch.no_grad():
                tmp = better_agent(nxt_observation).squeeze(dim = 0)
                tmp[action] += reward
                X.append(observation.squeeze(dim = 0))
                y.append(tmp)
    
    X_tensor = torch.stack(X)
    y_tensor = torch.stack(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    train_model(30, dataloader, better_agent, optimizer, lr, loss_function)


env.close()
