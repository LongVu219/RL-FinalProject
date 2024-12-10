import torch 
from actor_critic import ActorCritic

class PPO: 
    def __init__(
        self, 
        observation_shape, 
        action_shape, 
        episodes: int, 
        batch_size: int, 
        lr_actor: float, 
        lr_critic: float, 
        epsilon_clip: float,  
        epochs: int, 
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    ): 
        self.observation_shape = observation_shape 
        self.action_shape = action_shape 
        self.episodes = episodes
        self.batch_size = batch_size  
        self.lr_actor = lr_actor 
        self.lr_critic = lr_critic 
        self.epsilon_clip = epsilon_clip 
        self.epochs = epochs 
        self.device = device 

        self.policy = ActorCritic(
            observation_shape = self.observation_shape, 
            action_shape = self.action_shape
        ).to(self.device) 

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        self.policy_old = ActorCritic(
            observation_shape = self.observation_shape, 
            action_shape = self.action_shape
        ).to(self.device) 

        self.policy_old.load_state_dict(self.policy.state_dict()) 

        self.critic_loss = torch.nn.MSELoss() 
    

    # old netword has to select random action for update current network 

    
    def train_epoch(self, episode, epoch, dataloader: torch.utils.data.DataLoader): 
        losses = []
        for i, data in enumerate(dataloader): 
            old_states = data['old_states'].to(self.device).float()
            old_actions = data['old_actions'].to(self.device).float()
            old_log_probs = data['old_log_probs'].to(self.device).float()
            old_state_values = data['old_state_values'].to(self.device).float()
            returns = data['returns'].to(self.device).float()

            advantages = returns.detach() - old_state_values.detach() 

            current_log_probs, current_state_values, distribution_entropy = self.policy.evaluate(state=old_states, action=old_actions) 

            current_log_probs = current_log_probs.float() 
            current_state_values = current_state_values.float() 
            distribution_entropy = distribution_entropy.float()

            current_state_values = torch.squeeze(current_state_values) 

            ratios = torch.exp(current_log_probs - old_log_probs.detach()) 
            surr1 = ratios * advantages 
            surr2 = torch.clamp(ratios, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages 
            
            loss = torch.mean(-torch.min(surr1, surr2) + 0.5 * self.critic_loss(current_state_values, returns) - 0.01 * distribution_entropy)
            loss = loss.float() 

            self.optimizer.zero_grad() 
            
            loss.backward() 
            self.optimizer.step() 

            losses.append(loss.item())

            if i % 10 == 0: 
                self.policy_old.load_state_dict(self.policy.state_dict()) 
                print(f"Avg loss at episode {episode}, epoch {epoch}, time step {i}: {torch.mean(torch.tensor(losses))}")
        
    def train(self, episode, dataloader: torch.utils.data.DataLoader): 
        for epoch in range(1, self.epochs): 
            self.train_epoch(episode, epoch, dataloader) 

    def save(self, path): 
        torch.save(self.policy_old.state_dict(), path) 
    
    def load(self, path): 
        self.policy.load_state_dict(torch.load(path)) 
        self.policy_old.load_state_dict(torch.load(path)) 
     