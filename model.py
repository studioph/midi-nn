import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# scale attributes to 0-1
class VelocityLSTM(nn.Module):
    def __init__(self, num_features: int, activation=None):
        super(VelocityLSTM, self).__init__()
        self.lstm_input_size = num_features
        self.lstm_output_size = num_features
        self.activation = activation
        self.target_size = 1 # predicting velocity
        self.lstm = nn.LSTM(num_features, num_features, batch_first=True)
        self.hidden = (torch.randn(1, 1, num_features),
                        torch.randn(1, 1, num_features))
        self.hidden2target = nn.Linear(num_features, self.target_size)

    def forward(self, batch):
        lstm_out, _ = self.lstm(batch)
        target_space = self.hidden2target(lstm_out)
        if self.activation != None:
            scores = self.activation(target_space.view(len(batch), -1))
            return scores
        else:
            return target_space.view(len(batch), -1)