import numpy as np
import torch

"""
A basic predictor that simply uses the velocity of the previous note as the prediction, 
or the average velocity of the whole sequence for every value.
"""
class BasicVelocityPredictor:
    def __init__(self, avg):
        self.avg = avg
        pass

    def predict(self, tensor):
        return torch.cat((tensor[-1:], tensor[:-1]))

    def predict_sample_avg(self, tensor):
        avg_velocity = torch.mean(tensor).item()
        return torch.tensor(np.full((len(tensor)), avg_velocity))
    
    def predict_whole_avg(self, tensor):
        return torch.tensor(np.full((len(tensor)), self.avg))