import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper 
import base_model 
from utils import * 
from ppo import PPO
from eval import *

class CustomDataset(Dataset): 
    def __init__(
        self, 
        states, 
        actions, 
        log_probs, 
        state_values, 
        returns
    ): 
        super(CustomDataset, self).__init__() 
        self.states = states 
        self.actions = actions 
        self.log_probs = log_probs 
        self.state_values = state_values 
        self.returns = returns 
    
    def __len__(self): 
        return len(self.old_actions) 

    def __getitem__(self, index):
        return {
            "states": self.states[index].float(), 
            "actions": self.actions[index].float(), 
            "log_probs": self.log_probs[index].float(), 
            "state_values": self.state_values[index].float(), 
            "returns": self.returns[index].float()
        }


class Trainer: 
    def __init__(
        self, 
        enviroment: OrderEnforcingWrapper,
        episodes: int, 
        gamma: float,
        red_model_path: str, 
        blue_model_path: str,  
        device: str
    ):
        self.enviroment = enviroment 
        self.episodes = episodes 
        self.gamma = gamma 
        self.red_model_path = red_model_path 
        self.blue_model_path = blue_model_path 
        self.device = device
 
        self.red_model = base_model.QNetwork(self.enviroment.observation_space('red_0').shape, self.enviroment.action_space("red_0").n).to(self.device) 
        self.red_model.load_state_dict(torch.load(self.red_model_path, weights_only=True, map_location=self.device)) 

        self.ppo = PPO(
            observation_shape = self.enviroment.observation_space("blue_0").shape, 
            action_shape = self.enviroment.action_space("blue_0").n, 
            episodes = self.episodes, 
            batch_size = 128, 
            lr_actor = 0.0003, 
            lr_critic = 0.001, 
            epsilon_clip = 0.2, 
            epochs = 50,
            device = "cuda"
        )

    def train(self): 

        for episode in range(1, self.episodes + 1): 

            buffer: dict[str, dict[str, list]] = {}
            states, actions, log_probs, state_values, returns = [], [], [], [], [] 

            self.enviroment.reset()

            print(f"Start collect data for training of episode {episode}.")

            for id, agent in enumerate(self.enviroment.agent_iter()): 

                if agent not in buffer.keys(): 
                    buffer[agent] = {
                        "states": [], 
                        "actions": [],
                        "log_probs": [], 
                        "state_values": [], 
                        "rewards": [],
                    }
                
                observation, reward, termination, truncation, info = self.enviroment.last()

                if not (termination or truncation):
                    agent_handle = agent.split("_")[0] 
                    
                    if agent_handle == "red": 
                        action = get_action(
                            enviroment = self.enviroment, 
                            episode = episode,
                            agent = agent, 
                            observation = observation, 
                            network = self.red_model, 
                            policy = "random"
                        )

                        self.enviroment.step(action) 

                    elif agent_handle == "blue": 
                        state = torch.tensor(observation).permute(2, 0, 1).float().unsqueeze(0).to(self.device)   # 1x5x13x13
                        action, log_prob_action, state_value = self.ppo.policy.select_action(state) 

                        
                        buffer[agent]["states"].append(state.unsqueeze(0)) #(5x13x13)
                        buffer[agent]["actions"].append(action)  # (1,)
                        buffer[agent]["log_probs"].append(log_prob_action) #(1,)
                        buffer[agent]["state_values"].append(state_value)  #(1,)
                        buffer[agent]["rewards"].append(float(reward))

                        self.enviroment.step(action) 
                else: 
                    buffer[agent]["rewards"].append(reward)
                    action = None 
                    self.enviroment.step(action) 

                

            for agent in buffer.keys(): 

                buffer[agent]["rewards"] = buffer[agent]["rewards"][1:len(buffer[agent]["states"]) + 1]

                states.extend(buffer[agent]["states"]) 
                actions.extend(buffer[agent]["actions"]) 
                log_probs.extend(buffer[agent]["log_probs"])
                state_values.extend(buffer[agent]["state_values"]) 

                
                return_value = 0
                for i, reward in enumerate(reversed(buffer[agent]["rewards"])):
                    if i == 0: 
                        return_value = reward 
                    else: 
                        return_value = self.gamma * return_value + reward 
                    
            
                    returns.insert(0, return_value) 
            
            print(f"Collection of data from episode {episode} is end, start training")
            
            dataset = CustomDataset(
                states = states, 
                actions = actions, 
                log_probs = log_probs, 
                state_values = state_values, 
                returns = returns
            )

            dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=2) 

            self.ppo.train(episode=episode, dataloader=dataloader)  
            self.ppo.save(path=self.blue_model_path) 

            if episode % 20 == 0: 

                evaluate(red_agent=self.red_model, blue_agent=self.ppo, rounds=1, debug=True)
            

                    

                






                    



                    




        

