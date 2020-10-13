import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# learning rate could be too big or too small
# scale attributes to 0-1
# batch data into fixed length sequences
class VelocityLSTM(nn.Module):
    def __init__(self, num_features):
        super(VelocityLSTM, self).__init__()
        self.lstm_input_size = num_features
        self.lstm_output_size = num_features
        # target size should be 1
        self.target_size = 128 # 128 possible velocities
        self.lstm = nn.LSTM(num_features, num_features)
        self.hidden = (torch.randn(1, 1, num_features),
                        torch.randn(1, 1, num_features))
        self.hidden2target = nn.Linear(num_features, self.target_size)

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)
        # target should be 100x1
        target_space = self.hidden2target(lstm_out.view(len(sequence), -1))
        # scaling output? weights may get really big
        # try relu or no activation
        scores = F.log_softmax(target_space, dim=1)
        return scores