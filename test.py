from magent2.environments import battle_v4 
import numpy as np 


enviroment = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-1, attack_penalty=-0.1, attack_opponent_reward=1.5,
max_cycles=300, extra_features=False, render_mode = "rgb_array")

count = []
count_dead = []

enviroment.reset()

for agent in enviroment.agent_iter(): 
     
    observation, reward, termination, truncation, info = enviroment.last() 
    if not (termination or truncation): 
        count.append(agent)
        action = enviroment.action_space(agent).sample()
    else: 
        action = None  
        count_dead.append(agent)
    enviroment.step(action) 

print(len(count)) 
print(len(count_dead))
print(162 * 300)

