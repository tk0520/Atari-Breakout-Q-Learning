from collections import deque
from typing import Any
import numpy as np 
import random
import torch

from agent.tree import SumTree
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

class PrioritizedReplayMemory():
    def __init__(self, agent, state_size, action_size, epsilon=1e-2, alpha=0.6, importance_sampling_correction=0.4):
        self.tree = SumTree(size=settings.MAX_MEMORY)
        self.agent = agent

        # PER params
        self.epsilon = epsilon
        self.alpha = alpha
        self.importance_sampling_correction = importance_sampling_correction
        self.max_priority = epsilon

        self.observations = np.zeros((settings.MAX_MEMORY, 4, 84, 84), dtype=np.uint8)  # 5.6GB
        self.next_observations = np.zeros((settings.MAX_MEMORY, 4, 84, 84), dtype=np.uint8)  # 5.6GB
        self.actions = np.zeros((settings.MAX_MEMORY, action_size), dtype=np.int32)
        self.rewards = np.zeros(settings.MAX_MEMORY, dtype=np.float32)
        self.terminated = np.zeros(settings.MAX_MEMORY, dtype=np.int32)

        self.count = 0
        self.real_size = 0
        self.size = settings.MAX_MEMORY

    def __len__(self):
        return self.real_size
    
    def store(self, experience):
        observation, action, reward, next_observation, terminated = experience
        self.tree.add(self.max_priority, self.count)

        self.observations[self.count] = observation
        self.actions[self.count] = action
        self.rewards[self.count] = reward
        self.next_observations[self.count] = next_observation
        self.terminated[self.count] = terminated

        # self.observations[self.count] = torch.as_tensor(observation)
        # self.actions[self.count] = torch.as_tensor(action)
        # self.rewards[self.count] = torch.as_tensor(reward)
        # self.next_observations[self.count] = torch.as_tensor(next_observation)
        # self.terminated[self.count] = torch.as_tensor(terminated)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)
    
    def get_samples(self):
        assert self.real_size >= settings.BATCH_SIZE, "buffer contains less samples than batch size"

        sample_indices, tree_indices = [], []
        priorities = np.zeros((settings.BATCH_SIZE, 1))
        segment = self.tree.total / settings.BATCH_SIZE

        for i in range(settings.BATCH_SIZE):
            a, b = segment * i, segment * (i + 1)
            cumulated_sum = random.uniform(a, b)

            tree_idx, priority, sample_idx = self.tree.get(cumulated_sum)
           
            tree_indices.append(tree_idx)
            sample_indices.append(sample_idx)
            priorities[i] = priority

        
        probas = priorities / self.tree.total

        weights = (self.real_size * probas) ** -self.importance_sampling_correction
        weights = weights / weights.max()
        
        batch = (
                self.observations[sample_indices],   # numpy 배열 [BATCH, 4, 84, 84]
                self.actions[sample_indices],        # numpy 배열 [BATCH, action_size]
                self.rewards[sample_indices],        # numpy 배열 [BATCH]
                self.next_observations[sample_indices],  # numpy 배열 [BATCH, 4, 84, 84]
                self.terminated[sample_indices]      # numpy 배열 [BATCH]
            )
        return batch, weights, tree_indices, sample_indices

    def update_priorities(self, sample_indices, loss_batch):
        if isinstance(loss_batch, torch.Tensor):
            priorities = loss_batch.detach().cpu().numpy()

        priorities = (np.abs(priorities) + self.epsilon) ** self.alpha
        
        for sample_idx, priority in zip(sample_indices, priorities):
            self.tree.update(sample_idx, priority)
            self.max_priority = max(self.max_priority, priority)

        # tree_total = self.root_node.value
        # iterations = settings.BATCH_SIZE
        # selected_experiences = []

        # for i in range(iterations):
        #     random_values = np.random.uniform(0, tree_total)
        #     selected_experience = self.retrieve(random_values, self.root_node).value
        #     selected_experiences.append(selected_experience)
        
        # return selected_experiences
    
    # def get_samples_indices(self):
    #     return random.choices(range(len(self.memory)), k=settings.BATCH_SIZE, weights=self.priorities)
    
    # def get_samples(self, sample_indices):
    #     # return np.array(self.memory)[sample_indices]