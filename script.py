import torch, utils, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VelocityLSTM
import numpy as np

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

model = VelocityLSTM(NUM_FEATURES, BATCH_SIZE)
model.cuda()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

def validate():
    print('Validating...')
    model.eval()
    losses = []
    for sequence, targets in zip(x_test, y_test):
        try:
            model.zero_grad()
            output= model(sequence.cuda())
            
            loss = loss_function(output,targets.cuda())
            pass
        except:
            e = sys.exc_info()[0]
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
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            scores = model(batch.cuda())

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(scores, targets.cuda())
            loss.backward()
            optimizer.step()
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