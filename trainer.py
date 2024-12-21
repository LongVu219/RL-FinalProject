import torch 
from torch import nn 
import random 
from torch.nn import functional as F
from data import * 
from q_network import * 
from eval import * 

from pettingzoo.utils.wrappers.order_enforcing import OrderEnforcingWrapper 

from tqdm import tqdm 
import copy 


def soft_update(target_network, q_network, tau):
    for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
        target_param.data.copy_(tau * q_param.data + (1 - tau) * target_param.data)

def train_step(
    q_network, 
    q_network_opponent, 
    target_network, 
    dataloader, 
    optimizer, 
    gamma, 
    device, 
    opponent_ratio
):
    total_loss = 0.0

    for states, actions, rewards, next_states, dones in dataloader:
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if random.random() < opponent_ratio:
                next_actions = q_network_opponent(next_states).argmax(dim=-1)
            else:
                next_actions = q_network(next_states).argmax(dim=-1)

            next_q_values = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        target_values = rewards + gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, target_values)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


class Trainer: 
    def __init__(
        self, 
        enviroment, 
        episodes: int = 300, 
        max_steps: int = 300, 
        capacity: int = 10000, 
        batch_size: int = 128, 
        lr: float = 1e-4, 
        gamma: float = 0.99, 
        update_target_steps: int = 5, 
        tau: float = 0.1, 
        device: str = "cuda", 
        epsilon_min: float = 0.1,
        path_blue: str = "./pretrained_model/blue.cross.pth",
        path_red: str = "./pretrained_model/red_cross.pth", 
        log_path: str = "./log.txt"
    ): 
        self.enviroment = enviroment 
        self.episodes = episodes 
        self.max_steps = max_steps 
        self.capacity = capacity 
        self.batch_size = batch_size 
        self.lr = lr 
        self.gamma = gamma 
        self.update_target_steps = update_target_steps 
        self.tau = tau 
        self.device = device
        self.epsilon_min = epsilon_min 
        self.path_blue = path_blue 
        self.path_red = path_red  
        self.log_path = log_path 

        self.enviroment.reset() 
        example_agent = list(self.enviroment.agents)[0]
        observation_shape = self.enviroment.observation_space(example_agent).shape
        action_shape = self.enviroment.action_space(example_agent).n

        self.red = MyQNetwork(observation_shape=observation_shape, action_shape=action_shape).to(self.device)
        self.red_target = MyQNetwork(observation_shape=observation_shape, action_shape=action_shape).to(self.device) 
        self.red_target.load_state_dict(self.red.state_dict()) 
        self.red_target.eval() 

        self.blue = MyQNetwork(observation_shape=observation_shape, action_shape=action_shape).to(self.device)
        self.blue_target = MyQNetwork(observation_shape=observation_shape, action_shape=action_shape).to(self.device) 
        self.blue_target.load_state_dict(self.blue.state_dict()) 
        self.blue_target.eval() 

        self.optimizer_red = torch.optim.Adam(params=self.red.parameters(), lr=self.lr) 
        self.optimizer_blue = torch.optim.Adam(params=self.blue.parameters(), lr=self.lr) 

        self.buffer_blue = Buffer(self.capacity)
        self.buffer_red = Buffer(self.capacity)
    
    def train(self): 
        avg_projection = 100 
        for episode in tqdm(range(self.episodes)): 
            observations = self.enviroment.reset()
            agent_rewards = {agent: 0 for agent in self.enviroment.agents}
            total_loss_blue, total_loss_red = 0.0, 0.0

            epsilon = max(self.epsilon_min, (0.1 ** (episode/self.episodes)))

            for step in range(self.max_steps): 
                actions = {} 
                for agent, obs in observations[0].items(): 
                    obs_tensor = torch.tensor(obs).permute(2, 0, 1).float().unsqueeze(0).to(self.device) 

                    if "blue" in agent: 
                        if random.random() < epsilon: 
                            action = self.enviroment.action_space(agent).sample()
                        else: 
                            with torch.no_grad(): 
                                q_values = self.blue(obs_tensor) 
                                action = torch.argmax(q_values, dim=-1).item() 

                    elif "red" in agent:
                        if random.random() < epsilon:
                            action = self.enviroment.action_space(agent).sample()
                        else:
                            with torch.no_grad():
                                q_values = self.red(obs_tensor)
                                action = torch.argmax(q_values, dim=-1).item() 
                    actions[agent] = action 
                
                next_observations, rewards, terminations, truncations, _ = self.enviroment.step(actions)
                # print(next_observations)

                for agent in self.enviroment.agents:

                    next_state = next_observations[agent] if not (terminations[agent] or truncations[agent]) else None
                    dones = True if (terminations[agent] or truncations[agent]) else False

                    if "blue" in agent:
                        self.buffer_blue.push(
                            observations[0][agent],
                            actions[agent],
                            rewards[agent], 
                            next_state,
                            dones,
                        )
                    else :
                        self.buffer_red.push(
                            observations[0][agent],
                            actions[agent],
                            rewards[agent],
                            next_state,
                            dones,
                        )
                    agent_rewards[agent] += rewards[agent]

                if (len(self.buffer_red) >= self.batch_size and len(self.buffer_blue) >= self.batch_size):
                    red_dataloader = create_dataloader(self.buffer_red, self.batch_size)
                    blue_dataloader = create_dataloader(self.buffer_blue, self.batch_size)

                    # Train agents
                    loss_blue = train_step(self.blue, self.red, self.blue_target, blue_dataloader, self.optimizer_blue, self.gamma, self.device, opponent_ratio=0.8)
                    loss_red =  train_step(self.red, self.blue, self.red_target, red_dataloader, self.optimizer_red, self.gamma, self.device, opponent_ratio=0.)

                    total_loss_blue += loss_blue
                    total_loss_red += loss_red

                if step % self.update_target_steps == 0:
                    soft_update(self.red_target, self.red, self.tau)
                    soft_update(self.blue_target, self.blue, self.tau)

                observations = [copy.deepcopy(next_observations)]
                if (step + 1) % 50 == 0 or step == 0: 
                    print(f"For step: {step + 1}, Number of agent: {self.enviroment.num_agents}")
                if(self.enviroment.num_agents == 0): break


            blue_rewards = sum(reward for agent, reward in agent_rewards.items() if "blue" in agent)
            red_rewards = sum(reward for agent, reward in agent_rewards.items() if "red" in agent)
            avg_loss_blue = total_loss_blue / self.max_steps if len(self.buffer_blue) >= self.batch_size else 0
            avg_loss_red = total_loss_red / self.max_steps if len(self.buffer_red) >= self.batch_size else 0
        
            print(f"Episode {episode + 1}/{self.episodes}, Blue Reward: {blue_rewards}, Red Reward: {red_rewards}, "
                f"Blue Loss: {avg_loss_blue:.4f}, Red Loss: {avg_loss_red:.4f}, Epsilon: {epsilon:.2f}")
        
            with open(self.log_path, "a") as file: 
                file.write(f"Episode {episode + 1}/{self.episodes}, Blue Reward: {blue_rewards}, Red Reward: {red_rewards}, Blue Loss: {avg_loss_blue:.4f}, Red Loss: {avg_loss_red:.4f}, Epsilon: {epsilon:.2f} \n")
                file.close()

            avg = evaluate(None, blue_agent=self.blue, policy="best", debug = True, rounds=3)
            if avg < avg_projection: 
                torch.save(self.blue.state_dict(), self.path_blue)
                torch.save(self.red.state_dict(), self.path_red)
                avg_projection = avg 
            with open(self.log_path, "a") as file: 
                file.write(f"Avg projection power at episode {episode + 1}: {avg} \n")
                file.close()

            
        self.enviroment.close()
        
        print("Training completed and models saved!")