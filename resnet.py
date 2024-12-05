import torch.nn as nn
import torch


class ResBlock(nn.Module):
    def __init__(self, observation_shape, padding = 1):
        super().__init__()


        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1] * 2, 3, padding=padding),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1] * 2, observation_shape[-1], 1, padding=0),
            nn.ReLU()
        )
        
    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        final = x + self.cnn(x)
        return final

class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.resnet = nn.Sequential(
           ResBlock(observation_shape),
           ResBlock(observation_shape, padding=1),
           nn.Flatten()
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)

        dummy_output = self.resnet(dummy_input)
        flatten_dim = dummy_output.reshape(-1).shape[0]

        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.resnet(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)

