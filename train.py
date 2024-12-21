from trainer import Trainer 
from magent2.environments import battle_v4

enviroment = battle_v4.parallel_env(
    map_size=45,
    minimap_mode=False,
    step_reward=-0.005,
    dead_penalty=-0.1,
    attack_penalty=-0.05,
    attack_opponent_reward=0.5,
    extra_features=False,
    max_cycles=300,
)

episodes = 15
max_steps = 300 
capacity = 10000 
batch_size = 128 
lr = 1e-4
gamma = 0.99 
update_target_steps= 10
tau = 0.1
device = "cuda" 
epsilon_min = 0.1
path_blue = "./pretrained_model/blue_cross.pth"
path_red = "./pretrained_model/red_cross.pth"
log_path = "./log.txt"

trainer = Trainer(
    enviroment = enviroment, 
    episodes = episodes, 
    max_steps = max_steps, 
    capacity = capacity, 
    batch_size = batch_size, 
    lr = lr, 
    gamma = gamma, 
    update_target_steps = update_target_steps, 
    tau = tau, 
    device = device, 
    epsilon_min = epsilon_min, 
    path_blue = path_blue, 
    path_red = path_red, 
    log_path = log_path
)

trainer.train() 