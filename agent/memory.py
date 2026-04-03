from dataclasses import dataclass, field
from queue import PriorityQueue
from collections import deque
from typing import Any
import numpy as np 
import random
import torch

import settings

class ReplayMemory:
    def __init__(self):
        self.memory = deque(maxlen=settings.MAX_MEMORY)
    
    def store(self, experience):
        self.memory.append(experience)
    
    def get_samples(self):
        return random.sample(self.memory, settings.BATCH_SIZE)
    
    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, agent):
        super().__init__()
        self.priorities = deque(maxlen=settings.MAX_MEMORY)
        self.agent = agent

    def store(self, experience):
        observation, action, reward, next_observation, terminated = experience
        observation, next_observation = np.expand_dims(observation, 0), np.expand_dims(next_observation, 0)

        with torch.no_grad():
            current_q_values = self.agent.current_model.forward(observation).squeeze(0)
            current_q_value = current_q_values[action]

            target_q_values = self.agent.target_model.forward(next_observation).squeeze(0)
            target_q_value = target_q_values.max()

        td_error = reward + (1 - terminated) * settings.DISCOUNT * target_q_value - current_q_value
        td_error = abs(float(td_error.detach())) + 1e-6
        
        self.priorities.append(td_error)
        self.memory.append(experience)
    
    def get_samples(self):
        return random.choices(self.memory, weights=self.priorities, k=settings.BATCH_SIZE)


