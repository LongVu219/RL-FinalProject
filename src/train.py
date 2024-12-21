from magent2.environments import battle_v4
from trainer import Trainer 

enviroment = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
    dead_penalty=-1, attack_penalty=-0.1, attack_opponent_reward=1.5,
    max_cycles=300, extra_features=False, render_mode = "rgb_array")

trainer = Trainer(
    enviroment = enviroment, 
    episodes = 300, 
    gamma = 1.0, 
    red_model_path = "/mnt/apple/k66/hanh/RL-FinalProject/model/red.pt", 
    blue_model_path = "/mnt/apple/k66/hanh/RL-FinalProject/model/blue_random.pt", 
    policy = "random",
    device = "cuda"
)

trainer.train() 