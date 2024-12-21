import torch 
from torch import nn 
from torch import nn 
from res_net import ResBlock
from magent2.environments import battle_v4 
from utils import *  
from tqdm import tqdm 
import os 
import cv2 


device = 'cuda'

class CrossQNetwork(nn.Module): 
    def __init__(
        self, 
        observation_shape, 
        action_shape, 
    ): 
        super(CrossQNetwork, self).__init__() 
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

class BaseQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        return self.network(x)

class FinalQNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
            nn.Conv2d(observation_shape[-1], observation_shape[-1], 3),
            nn.ReLU(),
        )
        dummy_input = torch.randn(observation_shape).permute(2, 0, 1)
        dummy_output = self.cnn(dummy_input)
        flatten_dim = dummy_output.view(-1).shape[0]
        self.network = nn.Sequential(
            nn.Linear(flatten_dim, 120),
            # nn.LayerNorm(120),
            nn.ReLU(),
            nn.Linear(120, 84),
            # nn.LayerNorm(84),
            nn.Tanh(),
        )
        self.last_layer = nn.Linear(84, action_shape)

    def forward(self, x):
        assert len(x.shape) >= 3, "only support magent input observation"
        x = self.cnn(x)
        if len(x.shape) == 3:
            batchsize = 1
        else:
            batchsize = x.shape[0]
        x = x.reshape(batchsize, -1)
        x = self.network(x)
        self.last_latent = x
        return self.last_layer(x)
    

final_model = FinalQNetwork((13, 13, 5), (21)).to("cuda") 
final_model.load_state_dict(torch.load("../pretrained_model/red_final.pt")) 

base_model_red = BaseQNetwork((13, 13, 5), (21)).to("cuda") 
base_model_red.load_state_dict(torch.load("../pretrained_model/red.pt"))

my_model = CrossQNetwork((13, 13, 5), (21)).to("cuda")
my_model.load_state_dict(torch.load("../pretrained_model/blue.pth")) 

demo_env = battle_v4.env(map_size=45, max_cycles=300, render_mode="rgb_array")
demo_env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 24
frames = []

red_cnt = 81
blue_cnt = 81
mem = {}
step = 0

for agent in demo_env.agent_iter():

    observation, reward, termination, truncation, info = demo_env.last()
    
    agent_handle = agent.split("_")[0]

    if (reward >= 4.5):
        if (agent_handle == 'blue'): red_cnt -= 1
        else: blue_cnt -= 1

    if termination or truncation:
        action = None  # this agent has died
    else:
        if agent_handle == "blue":
            action = get_action(demo_env, None, agent, observation, my_model, 'best')
        else:
            action = get_action(demo_env, None, agent, observation, final_model, 'best')
    
    demo_env.step(action)
    if (step == 162):
        frames.append(demo_env.render())
        # print(demo_env.render())
        step = 0
    step += 1

print(red_cnt, blue_cnt)

height, width, _ = frames[0].shape
out = cv2.VideoWriter(
    os.path.join(vid_dir, f"battle_vs_best.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
for frame in frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
out.release()
print("Done recording battle !!!")

demo_env.close()
