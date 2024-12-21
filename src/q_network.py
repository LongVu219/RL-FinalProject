import torch 
from torch import nn 
from res_net import ResBlock 

device = torch.device("cuda") 

class MyQNetwork(nn.Module): 
    def __init__(
        self, 
        observation_shape, 
        action_shape, 
    ): 
        super(MyQNetwork, self).__init__() 
        self.w, self.h, self.c = observation_shape 
        self.action_shape = action_shape 
        self.main_net = nn.Sequential(
            ResBlock(observation_shape=observation_shape), 
            nn.Flatten(),
            nn.Linear(in_features=self.w * self.h * self.c, out_features=128),
            nn.ReLU(), 
            nn.Linear(in_features=128, out_features=64), 
            nn.ReLU(), 
            nn.Linear(64, self.action_shape),
        )
    
    def forward(self, x): 
        return self.main_net(x) 

# x = torch.randn((4, 5, 13, 13)) 
# net = QNetwork(observation_shape=(13, 13, 5), action_shape=21) 
# print(net(x).shape)

