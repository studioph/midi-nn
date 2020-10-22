import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TODO
# scale attributes to 0-1

"""
An LSTM model that predicts the velocity of a MIDI note given:
    - The pitch of the note
    - The duration of the note

The model contains 2 layers - and LSTM layer and a linear layer that maps the hidden space to the target space.

The model can be used with or without an activation function.
"""
class VelocityLSTM(nn.Module):
    def __init__(self, num_features: int, activation=None):
        super(VelocityLSTM, self).__init__()
        self.activation = activation
        self.target_size = 1 # predicting velocity
        # LSTM shape is (batch_size, seq_len, num_features)
        self.lstm = nn.LSTM(num_features, num_features, batch_first=True)
        self.hidden2target = nn.Linear(num_features, self.target_size)

    def forward(self, batch):
        lstm_out, _ = self.lstm(batch)
        target_space = self.hidden2target(lstm_out)
        if self.activation == None:
            return target_space.view(len(batch), -1)
        else:
            scores = self.activation(target_space.view(len(batch), -1))
            return scores