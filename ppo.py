import torch 
from actor_critic import ActorCritic
from eval import *

class PPO: 
    def __init__(
        self, 
        observation_shape, 
        action_shape, 
        episodes: int, 
        batch_size: int, 
        lr: float, 
        epsilon_clip: float,  
        epochs: int, 
        log_path: str, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    ): 
        self.observation_shape = observation_shape 
        self.action_shape = action_shape 
        self.episodes = episodes
        self.batch_size = batch_size  
        self.lr = lr
        self.epsilon_clip = epsilon_clip 
        self.epochs = epochs 
        self.log_path = log_path 
        self.device = device 

        self.losses = []

        self.policy = ActorCritic(
            observation_shape = self.observation_shape, 
            action_shape = self.action_shape
        ).to(self.device) 

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        self.critic_loss = torch.nn.MSELoss() 
    

    # old netword has to select random action for update current network 

    
    def train_epoch(self, episode, epoch, dataloader: torch.utils.data.DataLoader): 
        
        for i, data in enumerate(dataloader): 
            states = data['states'].to(self.device).float()
            actions = data['actions'].to(self.device).float()
            old_log_probs = data['log_probs'].to(self.device).float()
            returns = data['returns'].to(self.device).float()
            old_state_values = data['state_values'].to(self.device).float() 

            current_action_probs = self.policy.actor(states) 
            current_dist = torch.distributions.Categorical(current_action_probs) 
            current_log_probs = current_dist.log_prob(actions).float()
            current_state_values = self.policy.critic(states).squeeze().float()
            advantages = (returns - old_state_values).detach() 
            entropy = current_dist.entropy() 

            ratios = torch.exp(current_log_probs - old_log_probs.detach())
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages 
            

            loss = (torch.mean(-torch.min(surr1, surr2) + 0.5 * self.critic_loss(current_state_values, returns) - 1e-4 * entropy)).float()

            self.optimizer.zero_grad() 
            loss.backward() 
            self.optimizer.step() 

            self.losses.append(loss.item())      

            if i == len(dataloader) - 1 and epoch % 25 == 0:  
                with open(self.log_path, "a") as file: 
                    file.write(f"Avg loss at episode {episode}, epoch {epoch}: {torch.mean(torch.tensor(self.losses)).item()}" + "\n")

                print(f"Avg loss at episode {episode}, epoch {epoch}: {torch.mean(torch.tensor(self.losses)).item()}")
            

        
    def train(self, episode, dataloader: torch.utils.data.DataLoader): 
        for epoch in range(1, self.epochs + 1): 
            self.train_epoch(episode, epoch, dataloader) 

    def save(self, path): 
        torch.save(self.policy.state_dict(), path) 
    
    def load(self, path): 
        self.policy.load_state_dict(torch.load(path)) 

     