import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class VelocityLSTM(nn.Module):
    # num_msgs is the number of messages in the input MIDI file
    def __init__(self, hidden_dim):
        super(VelocityLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        #hidden dimensions can be whatever we want, the 1st dimension is 2 becuase there are 2 attributes (pitch, time)
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden_dim, batch_first=True)

        #add a fully connected layer
        self.full = nn.Linear(hidden_dim, hidden_dim)

        # The linear layer that maps from hidden state space to results space
        self.final = nn.Linear(hidden_dim, 1)  # second arg needs to vary with number of notes in MIDI file
        self.hidden = self.init_hidden()

    def forward(self, msgs):
        lstm_out, self.hidden = self.lstm(msgs)
        last_hidden = self.hidden[0][-1]
        full = self.full(lstm_out)
        yhat = F.relu(self.final(full))
        return yhat.view(yhat.shape[0], yhat.shape[1])
    def init_hidden(self):
        h_0 = Variable(torch.cuda.FloatTensor(1, 1, self.hidden_dim).zero_())

        c_0 = Variable(torch.cuda.FloatTensor(1, 1, self.hidden_dim).zero_())
        return (h_0, c_0)

        