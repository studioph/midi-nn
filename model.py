import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VelocityLSTM(nn.module):
    def __init__(self, num_features):
        super(VelocityLSTM, self).__init__()
        self.lstm_input_size = num_features
        self.lstm_output_size = num_features
        self.target_size = 128 # 128 possible velocities
        self.lstm = nn.LSTM(num_features, num_features)
        self.hidden = (torch.randn(1, 1, num_features),
                        torch.randn(1, 1, num_features))
        self.hidden2target = nn.Linear(num_features, self.target_size)

    def forward(self, sequence):
        lstm_out, self.hidden = self.lstm(sequence, self.hidden)
        target_space = self.hidden2target(lstm_out.view(len(sequence), -1))
        scores = F.log_softmax(target_space, dim=1)
        return scores