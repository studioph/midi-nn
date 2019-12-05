print('importing libraries')
from mido import MidiFile
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random, matplotlib.pyplot as plt

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

#load data from numpy files
print('loading data')
x_train = np.load('./data/x_train.npy')
x_test = np.load('./data/x_test.npy')
y_train = np.load('./data/y_train.npy')
y_test = np.load('./data/y_test.npy')

print('creating model')
learning_rate = 0.001
model = VelocityLSTM(128).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_function = nn.MSELoss()


def validation():
    print('Validating..')
    yhat,  losses = [], []
    model.eval()
    for idx in [batch_size * x for x in range(int(len(x_test) / batch_size))]:
        data = Variable(torch.Tensor(x_test[idx:idx + batch_size])).cuda()
        target =  Variable(torch.FloatTensor(y_test[idx:idx + batch_size])).cuda()
        model.zero_grad()
        output= model(data)

        loss = loss_function(output,target)


        output = output.data.cpu().numpy()
        losses.append(loss.item())
        # target_data = target.data.cpu().numpy()

        yhat.append(output)

    return yhat, np.mean(losses)

print('beginning training loop')
#train
train_loss, val_loss, yhats = [],[],[]
epochs = range(10)
batch_size = 50

for epoch in epochs:
    print("EPOCH %d" % epoch)

    losses = []
    # how often to print some info to stdout
    print_every = batch_size * 10

    model.train()
    for idx in [batch_size * x for x in range(int(len(x_train) / batch_size))]:
        data = Variable(torch.Tensor(x_train[idx:idx + batch_size])).cuda()
        target =  Variable(torch.FloatTensor(y_train[idx:idx + batch_size])).cuda()

        output = model(data)

        # loss = criterion(output, data)
        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if idx % print_every == 0:
            # print the average loss of the last 10 batches
            print("Train epoch: {} [batch #{}, seq length {}]\tLoss: {:.6f}".format(
                epoch, idx / print_every, data.size()[1], np.mean(losses[-5:])))
    loss = np.mean(losses)
    train_loss.append(loss)
    print("epoch loss: " + str(loss))
    yhat, valloss = validation()
    val_loss.append(valloss)
    yhats.append(yhat)
    print ('val loss:' + str(valloss))
import matplotlib.pyplot as plt
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss ae64')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


print (np.min(train_loss), np.min(val_loss))
i = np.argmin(val_loss)
np.save('train_loss.npy', np.array(train_loss))
np.save('val_loss.npy', np.array(val_loss))
np.save('res.npy', np.array(yhats[i]))
np.save('realy.npy',np.array(y_test))
