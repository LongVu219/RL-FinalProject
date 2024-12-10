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
    
    def select_action(self, state: torch.Tensor): 

        #generate distribution pi(a|s_t)
        action_probs = self.actor(state) 
        dist = torch.distributions.Categorical(action_probs) 

        #random action from distribution pi(a_t|s_t) (exploit) 
        action = dist.sample() 

        #get the log(pi(a_t|s_t))
        action_log_prob = dist.log_prob(action)  

        #get th state value V(s_t)
        state_value = self.critic(state) 
        
        action = torch.squeeze(action).item()
        action_log_prob = torch.squeeze(action_log_prob).item() 
        state_value  = torch.squeeze(state_value).item() 

        return action, action_log_prob, state_value  # a_t, log(pi(a_t|s_t)), V(s_t) => dataset


    