import torch

from .model import DQNAtari

class EvaluationAgent:
    def __init__(self, model_file, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluation_model = DQNAtari(self.state_size , self.action_size)

    def load_evaluation_model(self, model_file):
        state_dict = torch.load(model_file, map_location=self.device)
        self.evaluation_model = self.evaluation_model.load_state_dict(state_dict)

    def act(self, observation):
        device = next(self.evaluation_model.parameters()).device
        
        observation = torch.from_numpy(observation).float()
        observation = observation.unsqueeze(0) 
        observation = observation.to(device)

        q_values = self.evaluation_model.forward(observation)
        return q_values.squeeze(0).argmax().item()
