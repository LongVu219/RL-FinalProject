from magent2.environments import battle_v4

enviroment = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.005,
dead_penalty=-1, attack_penalty=-0.1, attack_opponent_reward=0.5,
max_cycles=300, extra_features=False, render_mode = "rgb_array")

enviroment.reset()
agents = [] 
rewards = []

for id, agent in enumerate(enviroment.agent_iter()): 

    

    

    observation, reward, termination, truncation, info = enviroment.last()

    action = enviroment.action_space(agent).sample() 
    enviroment.step(action) 

    if agent == 'red_0':
        print(f"___________ {id} ___________")
        print(action) 
        print(observation.shape)
        print(reward) 
        print(termination, truncation, info)
    
    if id == 162 * 3 + 1: 
        break 