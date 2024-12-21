import torch 
from torch.utils.data import Dataset, DataLoader 
from collections import deque 

class Buffer: 
    def __init__(
        self, 
        capacity
    ): 
        self.buffer = deque(maxlen=capacity) 

    def push(self, state, action, reward, next_state, done): 

        if state is not None and next_state is not None:
            self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self): 
        return len(self.buffer)

class CustomDataset(Dataset): 
    def __init__(
        self, 
        buffer: Buffer
    ): 
        self.buffer = buffer 
    
    def __len__(self): 
        return len(self.buffer.buffer) 
    
    def __getitem__(self, index):
        state, action, reward, next_state, done = self.buffer.buffer[index]
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1)
        next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1) if next_state is not None else None
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        return state, action, reward, next_state, done

def create_dataloader(buffer, batch_size): 
    dataset = CustomDataset(buffer=buffer)  
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


