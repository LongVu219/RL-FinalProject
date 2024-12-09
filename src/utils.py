import torch 
import numpy as np 
import random 


def torch_to_numpy(tensor: torch.Tensor): 
    return tensor.detach().cpu().numpy() 

def convert_obs(array: np.ndarray): 
    return torch.Tensor(array).float().permute(2, 0, 1).unsqueeze(0) 

def get_action(enviroment, episode: int, agent, observation: np.ndarray, network: torch.nn.Module, policy: str): 
    if (policy == 'random'): 
        return enviroment.action_space(agent).sample()
    
    if (policy == 'epsilon'):
        eps = max(0.1, 0.9 - episode/300)
        rd = random.random()
        if (rd < eps): 
            return enviroment.action_space(agent).sample()
        
        observation = convert_obs(observation)
        with torch.no_grad():
            q_values = network(observation)
        return torch.argmax(q_values, dim=1).detach().cpu().numpy()[0]

    if (policy == 'best'):
        observation = convert_obs(observation)
        with torch.no_grad():
            q_values = network(observation)
        return torch.argmax(q_values, dim=1).detach().cpu().numpy()[0]