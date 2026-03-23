from torch.optim import Adam, RMSprop
import numpy as np
import random
import torch
import time

from agent.learner.q_learning import QLearner, NStepQLearner
from agent.memory import ReplayMemory
from agent.model import DQN, DQNAtari
import settings

class CartPoleAgent:
    def __init__(self, batch_size, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.state_size = state_size
        self.action_size = action_size

        self.current_model = DQN(state_size, action_size).to(device=self.device)
        self.target_model = DQN(state_size, action_size).to(device=self.device)
        # self.optimizer = Adam(self.current_model.parameters(), lr=settings.LEARNING_RATE)
        self.optimizer = RMSprop(self.current_model.parameters(), lr=settings.LEARNING_RATE, alpha=0.95)
        self.target_model.load_state_dict(self.current_model.state_dict())
        
        self.replay_memory = ReplayMemory()
        self.learner = QLearner(self)
        
        self.epsilon = settings.EPSILON_START
        # self.actions = list(range(action_size))

    def current_evaluate(self, observation):
        return self.current_model.forward(observation)
    
    def target_evaluate(self, observation):
        with torch.no_grad():
            return self.target_model.forward(observation)

    def update_target_model(self):
        self.target_model.load_state_dict(self.current_model.state_dict())

    def epsilon_decay(self):
        self.epsilon = max(settings.EPSILON_END, self.epsilon * settings.EPSILON_DECAY)

    def act_test(self, observation):
        device = next(self.current_model.parameters()).device
       
        observation = torch.from_numpy(observation).float()
        observation = observation.unsqueeze(0) 
        observation = observation.to(device)

        q_values = self.current_model.forward(observation)
        return q_values

    def act(self, observation_batch):
        if observation_batch.ndim == 3:
            observation_batch = np.expand_dims(observation_batch, axis=0)
            obs_tensor = torch.from_numpy(observation_batch).to(next(self.current_model.parameters()).device)
            q_values = self.current_model(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
            return actions
        
        p = np.random.rand(self.batch_size) 
        
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation_batch).to(next(self.current_model.parameters()).device)
            q_values = self.current_model(obs_tensor)
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        explore_mask = p < self.epsilon
        if np.any(explore_mask):
            actions[explore_mask] = np.random.randint(0, self.action_size, size=explore_mask.sum())
        
        return actions

class AtariAgent(CartPoleAgent):
    def __init__(self, batch_size, state_size, action_size):
        super().__init__(batch_size, state_size, action_size)
        self.current_model = DQNAtari(state_size, action_size).to(device=self.device)
        self.target_model = DQNAtari(state_size, action_size).to(device=self.device)

        self.optimizer = RMSprop(self.current_model.parameters(), lr=settings.LEARNING_RATE, alpha=0.95)
        self.update_target_model()