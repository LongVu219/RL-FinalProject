from magent2.environments import battle_v4
import torch.optim as optim
from torch import nn
import os
import numpy as np
import random
import sys
from torch.utils.data import DataLoader, TensorDataset
import time

def debug(var):
    print(var)
    sys.exit()

start_time = time.time()

env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-1, attack_penalty=-0.1, attack_opponent_reward=0.5,
max_cycles=300, extra_features=False, render_mode = "rgb_array")
num_agent = 162
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


#base model
base_q_network = base_model.QNetwork(
    env.observation_space("red_0").shape, env.action_space("red_0").n
).to(device)
base_q_network.load_state_dict(
    torch.load("model/red.pt", weights_only=True, map_location=device)
)

#current training model
better_agent = resnet.QNetwork(
    env.observation_space("blue_0").shape, env.action_space("blue_0").n
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

    debug = False
    if (num%100 == 0): debug = True
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
    if (debug == True):
        print(f'Epoch number {id} loss : {total_loss}')
    
    return total_loss

def train_model(num_epoch, dataloader, model, optimzer, lr, loss_fn):
    current_loss = 100
    while(current_loss > 0.001):
        for epoch in range(1, num_epoch + 1):
            current_loss = train_1_epoch(epoch, dataloader, model, optimzer, lr, loss_fn)
        
#===============================================================================================================



#==================================Training for 100 episodes============================================================
from evaluate_fight import *

best_score = 100
episodes = 300
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
        agent_id = int(agent.split("_")[1])
        if termination or truncation:
            action = None  # this agent has died
        else:
            if agent_handle == "red":
                action = get_action(env, episode, agent, observation, base_q_network, 'random')
            else:
                action = get_action(env, episode, agent, observation, better_agent, 'epsilon')
        
        if (agent not in buffer):
            buffer[agent] = []
        buffer[agent].append((agent, observation, action, reward, termination, truncation, info))
        env.step(action)

    for agent in buffer.keys():
        state_array = buffer[agent]
        agent_handle = agent.split("_")[0]
        #if (agent_handle == 'red'): continue

        for i in range(0, len(state_array)):
            agent, observation, action, prv_reward, termination, truncation, info = state_array[i]
            _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i]
            if (i < len(state_array) - 1):
                _1, nxt_observation, nxt_action, reward, nxt_termination, nxt_truncation, nxt_info = state_array[i + 1]

            observation = convert_obs(observation)
            nxt_observation = convert_obs(nxt_observation)

            with torch.no_grad():
                next_max = better_agent(nxt_observation).squeeze(dim = 0).max()
                tmp = better_agent(observation).squeeze(dim = 0)
                if (i == len(state_array)):
                    next_max = 0
                tmp[action] = reward + next_max
                X.append(observation.squeeze(dim = 0))
                y.append(tmp)


    X_tensor = torch.stack(X)
    y_tensor = torch.stack(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    #Train model with given data
    train_model(100, dataloader, better_agent, optimizer, lr, loss_function)

    if (episode % 5 == 0):
        avg = evaluate(red_agent=base_q_network, blue_agent=better_agent, rounds=5, debug = True)
        if (avg < best_score):
            torch.save(better_agent.state_dict(), 'model/agent_1.pth')
            print('Agent saved !')
            print()
            best_score = avg

print(best_score)
env.close()
#===================================================================================================================

end_time = time.time()

print(f'Total running time : {(end_time - start_time)/3600}hrs')

