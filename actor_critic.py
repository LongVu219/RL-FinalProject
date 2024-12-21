import torch 
from torch import nn
import random
from res_net import ResBlock


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape,
        action_shape
    ):
        super(ActorCritic, self).__init__() 
        self.observation_shape = observation_shape
        self.action_shape = action_shape

        self.actor = nn.Sequential(
            ResBlock(self.observation_shape),
            nn.Flatten(),
            nn.Linear(
                in_features=self.observation_shape[0]*self.observation_shape[1]*self.observation_shape[2], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64), 
            nn.ReLU(),
            nn.Linear(64, self.action_shape),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            ResBlock(self.observation_shape), 
            nn.Flatten(),
            nn.Linear(in_features=self.observation_shape[0]*self.observation_shape[1]*self.observation_shape[2], out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def select_action(self, state): 
        rd = random.random()
        with torch.no_grad(): 
            action_prob = self.actor(state) 
            action_dist = torch.distributions.Categorical(action_prob) 
            if rd <= 0.8:
                action = action_dist.sample()
            else: 
                action = torch.argmax(action_prob, dim=-1)
            
            action_log_prob = action_dist.log_prob(action) 
            state_value = self.critic(state) 

            return action.item(), action_log_prob.item(), state_value.item() 
