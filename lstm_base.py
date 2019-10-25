from mido import MidiFile
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
Data_dir = '/home/tug27634/Desktop/Music/midi_files/'
'Midi file processing'
def extract_attributes(mid):
    track = mid.tracks[0] #assuming single track MIDI file in this scenario, as all of the ones in the dataset are
    #filter for just the note_on messages
    note_on_msgs = [msg for msg in track if msg.type == 'note_on']
    #convert the message into a matrix of [pitch, time]
    data = [[msg.note, msg.time] for msg in note_on_msgs]
    return data
def get_velocities(mid):
    track = mid.tracks[0] #assuming single track MIDI file in this scenario, as all of the ones in the dataset are
    #filter for just the note_on messages
    note_on_msgs = [msg for msg in track if msg.type == 'note_on']
    data = [msg.velocity for msg in note_on_msgs]
    return data

def loadMidi(path):
    # can remove if check if more RAM is present when loading all the files
    files = [file for file in os.listdir(path) if 'transpose' not in file]
    # files.remove('desktop.ini')  # gets caught in listdir() call on Windows

    # load the files using mido
    midis = [MidiFile(path + file) for file in files]  # using 10 files to start
    x = [extract_attributes(mid) for mid in midis]
    y = [get_velocities(mid) for mid in midis]
    return (x, y)
'x, y are two 100 length lists. Each elements contains all notes in its midi file. For example, len(x[0]) = 3632 means' \
'it has 3632 notes. However, our idea is concatenate all notes together and split with every 100 notes '
x, y = loadMidi(Data_dir)
'Because of tutorial, I just make data into simple. We only care about first 2 midi files'
x = x[:2]; y = y[:2]
'Concatenate x and y'
x_concat = []; y_concat = []
for idx in range(2):
    for note in x[idx]:
        x_concat.append(note)
    for veloc in y[idx]:
        y_concat.append(veloc)
print (len(x_concat), len(y_concat))
'Now, we have 16930 notes in total. 16930/100 = 169.3 we can generate 169 files'
'Resplit into 100 notes for each'
def resplit(x_concat, y_concat):
    new_x = []; new_y = []
    count = int(len(x_concat)/100)
    for idx in range(count):
        new_x.append(np.array(x_concat[idx*100:(idx+1)*100]))
        new_y.append(np.array(y_concat[idx * 100:(idx + 1) * 100]))

    return new_x, new_y



new_x, new_y = resplit(x_concat, y_concat)

'Split train, test'
import random

def train_test_split(new_x, new_y, percent):
    random.seed(1)
    idxlist = range(len(new_x))
    random.shuffle(idxlist)
    train_id = idxlist[:int(len(idxlist)*percent)]
    test_id = idxlist[int(len(idxlist)*percent):]
    return [new_x[idx] for idx in train_id],[new_y[idx] for idx in train_id],[new_x[idx] for idx in test_id],[new_y[idx] for idx in test_id]


x_train, y_train, x_test, y_test = train_test_split(new_x, new_y, 0.8)


class VelocityLSTM(nn.Module):
    # num_msgs is the number of messages in the input MIDI file
    def __init__(self, hidden_dim):
        super(VelocityLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(2, hidden_dim)
        #         self.embeddings = nn.Embedding(num_msgs, embedding_dim)

        # The linear layer that maps from hidden state space to results space
        self.final = nn.Linear(hidden_dim, 1)  # second arg needs to vary with number of notes in MIDI file
        self.hidden = self.init_hidden()

    def forward(self, msgs):
        lstm_out, self.hidden = self.lstm(msgs.view(msgs.shape[0], 1, -1))
        last_hidden = self.hidden[0][-1]

        # yhat = F.relu(self.final(last_hidden))
        yhat = F.relu(self.final(lstm_out))
        return yhat.view(yhat.shape[0])
    def init_hidden(self):
        h_0 = Variable(torch.cuda.FloatTensor(1, 1, self.hidden_dim).zero_())

        c_0 = Variable(torch.cuda.FloatTensor(1, 1, self.hidden_dim).zero_())
        return (h_0, c_0)



learning_rate = 0.001
model = VelocityLSTM(8).cuda()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_function = nn.MSELoss()


def validation():
    print('Validating..')
    y, yhat,  losses = [], [], []
    model.eval()
    for idx, data in enumerate(x_test):
        data = np.array(data)
        data = Variable(torch.Tensor(data)).cuda()
        target =  Variable(torch.FloatTensor(np.array(y_test[idx]))).cuda()
        model.zero_grad()
        output= model(data)

        loss = loss_function(output,target)


        output = output.data.cpu().numpy()
        losses.append(loss.item())
        target_data = target.data.cpu().numpy()

        yhat.append(output)

    return yhat, np.mean(losses)
train_loss, val_loss, yhats = [],[],[]
for epoch in range(200):
    print("EPoCH %d" % epoch)

    losses = []
    # how often to print some info to stdout
    print_every = 10

    model.train()
    for idx, data in enumerate(x_train):
        data = np.array(data)
        data = Variable(torch.Tensor(data)).cuda()
        target =  Variable(torch.FloatTensor(np.array(y_train[idx]))).cuda()

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
                epoch, idx, data.size()[0], np.mean(losses[-5:])))
    loss = np.mean(losses)
    train_loss.append(loss)
    print("epoch loss: " + str(loss))
    yhat, valloss = validation()
    val_loss.append(valloss)
    yhats.append(yhat)
    print ('val loss:' + str(valloss))
import matplotlib.pyplot as plt
epochs = range(200)
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss ae64')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


print np.min(train_loss), np.min(val_loss)
i = np.argmin(val_loss)
np.save('train_loss.npy', np.array(train_loss))
np.save('val_loss.npy', np.array(val_loss))
np.save('res.npy', np.array(yhats[i]))
np.save('realy.npy',np.array(y_test))
