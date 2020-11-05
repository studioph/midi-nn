import torch, sys
import midi_nn.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from midi_nn.model import VelocityLSTM
import numpy as np
from datetime import datetime
from midi_nn.dataset import MIDIDataset

# TODO
# warm-up for first 10 notes for example - feed predicted after first N notes

utils.checkGPU() # throws an error if the GPU isn't detected

####################
# define constants
####################
TRAIN_FILE = 'data/train.npy'
TEST_FILE = 'data/test.npy'
NUM_FEATURES = 6
BATCH_SIZE = 64
SEQ_LENGTH = 100
NUM_EPOCHS = 100
MODEL_SAVE_FILE = 'model.zip'
LOSS_SAVE_DIR = 'losses'
LEARNING_RATE = 1e-5

##################################
# define Datasets and Dataloaders
##################################
train_dataset = MIDIDataset(TRAIN_FILE)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = MIDIDataset(TEST_FILE)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

###################################
# define model and hyperparameters
###################################
model = VelocityLSTM(NUM_FEATURES, F.relu)
model.cuda()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
min_loss = sys.maxsize

"""
Runs the validation loop on the model

Returns:
    mean_loss (float) - The average loss of all the validation samples
"""
def validate():
    print('Validating...')
    model.eval()
    losses = []
    for batch, targets in test_loader:
        model.zero_grad()
        output= model(batch.cuda())
        loss = loss_function(output,targets.cuda())
        losses.append(loss.item())
    mean_loss = np.mean(losses)
    # getting undefined variable error (min_loss) here for some reason even though it is defined above
    # if mean_loss < min_loss:
    #     min_loss = mean_loss
    #     torch.save(model, MODEL_SAVE_FILE)
    print(f'Loss: {round(mean_loss, 4)}')
    return mean_loss

"""
Runs the training loop on the model, validating at the end of every epoch.

Returns:
    (train_losses, test_losses) - A tuple containing both the training and validation losses as arrays
"""
def train():
    train_losses = []
    test_losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_losses = []
        print(f'Training epoch {epoch + 1}...')
        for batch, targets in train_loader:
            model.zero_grad()
            scores = model(batch.cuda())
            loss = loss_function(scores, targets.cuda())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_losses.append(np.mean(epoch_losses))
        test_losses.append(validate())
    print('Training complete')
    return train_losses, test_losses

train_losses, test_losses = train()

###########################################
# Plot losses and save the model to a file
###########################################
utils.plot_losses(train_losses, test_losses, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE)
torch.save(model, MODEL_SAVE_FILE)

###########################################
# See what final scores are after training
###########################################
print('Final scores after training:')
with torch.no_grad():
    inputs, targets = next(iter(train_loader))
    scores = model(inputs.cuda())
    print(scores)