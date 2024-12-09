import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper 
import base_model 
from utils import * 
from ppo import PPO

class CustomDataset(Dataset): 
    def __init__(
        self, 
        old_states, 
        old_actions, 
        old_log_probs, 
        old_state_values, 
        returns
    ): 
        super(CustomDataset, self).__init__() 
        self.old_states = old_states 
        self.old_actions = old_actions 
        self.old_log_probs = old_log_probs 
        self.old_state_values = old_state_values 
        self.returns = returns 
    
    def __len__(self): 
        return len(self.old_actions) 

    def __getitem__(self, index):
        return {
            "old_states": self.old_states[index], 
            "old_actions": self.old_actions[index], 
            "old_log_probs": self.old_log_probs[index], 
            "old_state_values": self.old_state_values[index], 
            "returns": self.returns[index]
        }


class Trainer: 
    def __init__(
        self, 
        enviroment: OrderEnforcingWrapper,
        episodes: int, 
        gamma: float,
        red_model_path: str, 
        blue_model_path: str,  
    ):
        self.enviroment = enviroment 
        self.episodes = episodes 
        self.gamma = gamma 
        self.red_model_path = red_model_path 
        self.blue_model_path = blue_model_path 
 
        self.red_model = base_model.QNetwork(self.enviroment.observation_space('red_0').shape, self.enviroment.action_space("red_0").n).to(self.device) 
        self.red_model.load_state_dict(torch.load(self.red_model_path, weights_only=True, map_location=self.device)) 

        self.ppo = PPO(
            observation_shape = enviroment.observation_space("blue_0").shape, 
            action_shape = enviroment.action_space("blue_0").shape, 
            episodes = episodes, 
            batch_size = 128, 
            lr_actor = 1e-5, 
            lr_critic = 1e-5, 
            gamma = 1.0, 
            epsilon_clip = 0.2, 
            epochs = 50 
        )

    def train(self): 

        for episode in range(1, self.episodes + 1): 

            buffer: dict[str, dict[str, list]] = {}

            old_states, old_actions, old_log_probs, old_state_values, returns = [], [], [], [], [] 
            self.enviroment.reset()

            for id, agent in enumerate(self.enviroment.agent_iter()): 

                if agent not in buffer.keys(): 
                    buffer[agent] = {
                        "old_states": [], 
                        "old_actions": [],
                        "old_log_probs": [], 
                        "old_state_values": [], 
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
                        old_state = torch.tensor(observation).float().unsqueeze(0)       # 13 x 13 x 5 => 1 x 13 x 13 x 5
                        old_action, old_log_prob, old_state_value = self.ppo.policy_old.act(old_state)
                        
                        buffer[agent]["old_states"].append(old_state.squeeze(0))
                        buffer[agent]["old_actions"].append(old_action.squeeze(0)) 
                        buffer[agent]["old_log_probs"].append(old_log_prob.squeeze(0)) 
                        buffer[agent]["old_state_values"].append(old_state_value.squeeze(0))  
                        buffer[agent]["rewards"].append(reward)

                        self.enviroment.step(int(old_action.item())) 
                else: 
                    buffer[agent]["rewards"].append(reward)
                    action = None 
                    self.enviroment.step(action) 

            buffer[agent]["rewards"] = buffer[agent]["rewards"][0:len(buffer[agent]["old_actions"]) + 1]
    
            """
            after run this loop, for example
            
            """

            for agent in buffer.keys(): 
                old_states.extend(buffer[agent]["old_states"]) 
                old_actions.extend(buffer[agent]["old_actions"]) 
                old_log_probs.extend(buffer[agent]["old_log_probs"])
                old_state_values.extend(buffer[agent]["old_state_values"]) 
                
                return_value = 0
                for i, reward in enumerate(reversed(buffer[agent]["rewards"])):
                    if i == 0: 
                        return_value = reward 
                    else: 
                        return_value = self.gamma * return_value + reward 
                    
                    if i == len(buffer[agent]["rewards"]) - 1: 
                        break
                    else:
                        returns.insert(0, return_value) 
            

                    

                






                    



                    




        

