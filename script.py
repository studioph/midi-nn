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
from os import listdir
from data import load_data, sequence_data, batch_data, reconstruct, split
import random
from model import VelocityLSTM
import sys

batch_size = 10
seq_len = 100
load_dir = './data/numpy/'
save_dir = './data/results/'
midi_dir = './data/original/'

print('loading data')
data = load_data(load_dir, size_limit=100)
data = sequence_data(data, seq_len)
random.shuffle(data)
data = batch_data(data, batch_size)
train, test = split(data, 0.8)

print('creating model')
hidden_dim = 128
learning_rate = 0.001
model = VelocityLSTM(hidden_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_function = nn.MSELoss()

epochs = range(10)

def validation():
    print('Validating..')
    yhat,  losses, merged = [], [], []
    model.eval()
    for batch in test:
        data = Variable(torch.FloatTensor([row[:,[1,2]] for row in batch[:,1]])).cuda()
        target =  Variable(torch.FloatTensor([row[:,3] for row in batch[:,1]])).cuda()
        try:
            model.zero_grad()
            output= model(data)
            
            loss = loss_function(output,target)
            pass
        except:
            e = sys.exc_info()[0]
            pass


        output = output.data.cpu().numpy()
        merged.append((batch[:,0], [row[:,0] for row in batch[:,1]], output))
        losses.append(loss.item())
        # target_data = target.data.cpu().numpy()

        yhat.append(output)

    return yhat, np.mean(losses), merged

print('beginning training loop')
#train
train_loss, val_loss, yhats, merges = [],[],[], []
for epoch in epochs:
    print("EPOCH %d" % epoch)

    losses = []
    # how often to print some info to stdout
    print_every = 10

    model.train()
    for idx, batch in enumerate(train):
        # batch = np.array(batch)
        data = Variable(torch.FloatTensor([row[:,[1,2]] for row in batch[:,1]])).cuda()
        target =  Variable(torch.FloatTensor([row[:,3] for row in batch[:,1]])).cuda()

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
    yhat, valloss, merged = validation()
    val_loss.append(valloss)
    yhats.append(yhat)
    merges.append(merged)
    print ('val loss:' + str(valloss))


import matplotlib.pyplot as plt
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss ae64')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()


print (np.min(train_loss), np.min(val_loss))
i = np.argmin(val_loss)
print('reconstructing MIDI files')
reconstruct(merges[i][0], midi_dir, save_dir)
np.save('train_loss.npy', np.array(train_loss))
np.save('val_loss.npy', np.array(val_loss))
np.save('res.npy', np.array(yhats[i]))
# np.save('realy.npy',np.array(y_test))