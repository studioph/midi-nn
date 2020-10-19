import torch, utils, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VelocityLSTM
import numpy as np

# give real velocity of previous note, start at 2nd note? or give default value for first
# warm-up for first 10 notes for example - feed predicted after first N notes

NUM_FEATURES = 2
BATCH_SIZE = 64
SEQ_LENGTH = 100
NUM_EPOCHS = 10
MODEL_SAVE_FILE = 'model'
LOSS_SAVE_FILE = 'losses.npy'
SCORES_SAVE_FILE = 'scores.npy'

utils.checkGPU()

x_train, x_test, y_train, y_test = np.load('data/train_test.npy', allow_pickle=True)

# convert data arrays to tensors
y_train = [torch.tensor(batch).float() for batch in y_train]
y_test = [torch.tensor(batch).float() for batch in y_test]

x_train = [torch.tensor(batch).float() for batch in x_train]
x_test = [torch.tensor(batch).float() for batch in x_test]

model = VelocityLSTM(NUM_FEATURES, F.relu)
model.cuda()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def validate():
    print('Validating...')
    model.eval()
    losses = []
    for batch, targets in zip(x_test, y_test):
        try:
            model.zero_grad()
            output= model(batch.cuda())
            loss = loss_function(output,targets.cuda())
            pass
        except:
            e = sys.exc_info()[0]
            print(e)
            pass
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    print(f'Loss: {round(mean_loss, 4)}')
    return mean_loss

def train():
    # training loop
    losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        print(f'Training epoch {epoch + 1}...')
        for batch, targets in zip(x_train, y_train):
            try:
                model.zero_grad()
                scores = model(batch.cuda())
                loss = loss_function(scores, targets.cuda())
                loss.backward()
                optimizer.step()
                pass
            except:
                e = sys.exc_info()[0]
                print(e)
                pass
        losses.append(validate())
    print('Training complete')
    return losses

losses = train()
np.save(LOSS_SAVE_FILE, losses)
torch.save(model, MODEL_SAVE_FILE)

print('Final scores after training:')
with torch.no_grad():
    inputs = x_train[0].cuda()
    scores = model(inputs)
    print(scores)
    np.save(SCORES_SAVE_FILE, scores.cpu())