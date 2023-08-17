import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        if seed:
            random.seed(seed)
    
    def clear_buffer(self):
        self.memory = deque(maxlen=self.buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        sample_size = min(self.batch_size, len(self.memory))
        experiences = random.sample(self.memory, k=sample_size)
        
        # TODO: check torch.from_numpy when states from multiple episodes are present
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states = [torch.from_numpy(e.state).float().to(device) for e in experiences if e is not None]
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states = [torch.from_numpy(e.next_state).float().to(device) for e in experiences if e is not None]
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)