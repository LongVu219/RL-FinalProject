import torch 
from torch import nn 

class ResBlock(nn.Module):
    def __init__(
        self, 
        observation_shape 
    ): 
        super(ResBlock, self).__init__()
        self.observation_shape = observation_shape 
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=observation_shape[-1], out_channels=observation_shape[-1], kernel_size=3, padding=1), 
            nn.Conv2d(in_channels=observation_shape[-1], out_channels=observation_shape[-1]*2, kernel_size=3, padding=1), 
            nn.Conv2d(in_channels=observation_shape[-1]*2, out_channels=observation_shape[-1], kernel_size=3, padding=1),
            nn.Conv2d(in_channels=observation_shape[-1], out_channels=observation_shape[-1], kernel_size=1, padding=0),  
        )
    def forward(self, x): 
        return x + self.main(x) 