import torch 
from torch import nn 

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
            ResBlock(self.observation_shape), 
            nn.Flatten(),
            nn.Linear(in_features=self.observation_shape[0]*self.observation_shape[1]*self.observation_shape[2], out_features=128),
            nn.ReLU(), 
            nn.Linear(in_features=128, out_features=64), 
            nn.ReLU(), 
            nn.Linear(64, self.action_shape),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            ResBlock(self.observation_shape), 
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


    def act(self, state: torch.Tensor): # sample action from old network and return (action, action_log_probs, state_value) of old network 
        action_probs = self.actor(state) 
        dist = torch.distributions.Categorical(action_probs) 

        action = dist.sample() 
        action_log_probs = dist.log_prob(action) 

        state_value = self.critic(state) 

        return action.detach(), action_log_probs.detach(), state_value.detach()
    
    # return (action_log_probs, state_value, entropy loss of distribution) of current network with (old state and old action)  

    def evaluate(self, state: torch.Tensor, action: torch.Tensor): 
        action_probs = self.actor(state) 
        dist = torch.distributions.Categorical(action_probs) 
        
        action_log_probs = dist.log_prob(action) 
        dist_entropy = dist.entropy() 
        state_value = self.critic(state) 

        return action_log_probs, state_value, dist_entropy 