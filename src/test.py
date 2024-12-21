import torch 
from torch import nn 
from torch import nn 
from res_net import ResBlock
from magent2.environments import battle_v4 
from utils import *  
from tqdm import tqdm 


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
final_model.load_state_dict(torch.load("/mnt/apple/k66/hanh/cross_q/pretrained_model/red_final.pt")) 

base_model_red = BaseQNetwork((13, 13, 5), (21)).to("cuda") 
base_model_red.load_state_dict(torch.load("/mnt/apple/k66/hanh/cross_q/pretrained_model/red.pt"))

my_model = CrossQNetwork((13, 13, 5), (21)).to("cuda") 
my_model.load_state_dict(torch.load("/mnt/apple/k66/hanh/cross_q/pretrained_model/blue.pth")) 


def eval(red, blue, red_policy, blue_policy, episodes): 
    env = battle_v4.env(map_size=45, max_cycles=300)
    env.reset()

    red_win, blue_win = [], [] 
    red_tot_rw, blue_tot_rw = [], []
    n_agent_each_team = len(env.env.action_spaces) // 2

    for _ in tqdm(range(episodes)):
        env.reset()
        n_kill = {"red": 0, "blue": 0}
        red_reward, blue_reward = 0, 0

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            agent_team = agent.split("_")[0]

            n_kill[agent_team] += (
                reward > 4.5
            )  # This assumes default reward settups
            if agent_team == "red":
                red_reward += reward
            else:
                blue_reward += reward

            if termination or truncation:
                action = None  # this agent has died
            else:
                if agent_team == "red":
                    action = get_action(env, None, agent, observation, red, red_policy)
                else:
                    action = get_action(env, None, agent, observation, blue, blue_policy)

            env.step(action)

        who_wins = "red" if n_kill["red"] >= n_kill["blue"] + 5 else "draw"
        who_wins = "blue" if n_kill["red"] + 5 <= n_kill["blue"] else who_wins
        red_win.append(who_wins == "red")
        blue_win.append(who_wins == "blue")

        red_tot_rw.append(red_reward / n_agent_each_team)
        blue_tot_rw.append(blue_reward / n_agent_each_team)

    return {
        "winrate_red": np.mean(red_win),
        "winrate_blue": np.mean(blue_win),
        "average_rewards_red": np.mean(red_tot_rw),
        "average_rewards_blue": np.mean(blue_tot_rw),
    }



if __name__ == "__main__": 
    print("----------------Play with red.pt------------------ \n")
    print(eval(red=base_model_red, blue=my_model, red_policy="best", blue_policy="best", episodes=100))
    print("----------------Play with final_red.pt------------------ \n")
    print(eval(red=final_model, blue=my_model, red_policy="best", blue_policy="best", episodes=100))




    
    