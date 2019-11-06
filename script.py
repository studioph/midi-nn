from data import load_data 
from model import VelocityLSTM
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# load data
print('loading data')
data_dir = './data/numpy/'
x_train, x_test, y_train, y_test = load_data(data_dir)
x_train = x_train[2]
y_train = y_train[2]

print('creating model')
hidden_dim = 128
learning_rate = 0.001
model = VelocityLSTM(hidden_dim).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_function = nn.MSELoss()

def validation():
    print('Validating..')
    y, yhat, metadata, losses = [], [], [], []
    model.eval()
    for idx in [batch_size * x for x in range(int(len(x_test[2]) / batch_size))]:
        data = Variable(torch.Tensor(x_test[2][idx:idx + batch_size])).cuda()
        metadata.append(x_test[:2][idx:idx + batch_size])
        target =  Variable(torch.FloatTensor(y_test[2][idx:idx + batch_size])).cuda()
        model.zero_grad()
        output= model(data)

        loss = loss_function(output,target)


        output = output.data.cpu().numpy()
        losses.append(loss.item())
        target_data = target.data.cpu().numpy()

        yhat.append(output)

    return yhat, np.mean(losses), metadata

print('beginning training loop')
#train
train_loss, val_loss, yhats, metadatas = [],[],[], []
epochs = range(100)
batch_size = 10

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
    yhat, valloss, metadata = validation()
    val_loss.append(valloss)
    metadatas.append(metadata)
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
np.save('meta.npy', np.array(metadatas[i]))
np.save('realy.npy',np.array(y_test))