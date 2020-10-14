import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# learning rate could be too big or too small
# scale attributes to 0-1
# batch data into fixed length sequences
class VelocityLSTM(nn.Module):
    def __init__(self, num_features, batch_size):
        super(VelocityLSTM, self).__init__()
        self.lstm_input_size = num_features
        self.lstm_output_size = num_features
        self.batch_size = batch_size
        self.num_features = num_features
        self.target_size = 1 # predicting velocity
        self.lstm = nn.LSTM(num_features, num_features, batch_first=True)
        self.hidden = (torch.randn(1, 1, num_features),
                        torch.randn(1, 1, num_features))
        self.hidden2target = nn.Linear(num_features, self.target_size)

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence)
        # target should be 100x1
        target_space = self.hidden2target(lstm_out)
        # scaling output? weights may get really big
        # try relu or no activation
        scores = F.relu(target_space.view(self.batch_size, -1))
        return scores