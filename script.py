import torch, utils, sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VelocityLSTM
import numpy as np
from dataset import MIDIDataset

# give real velocity of previous note, start at 2nd note? or give default value for first
# warm-up for first 10 notes for example - feed predicted after first N notes
# randomize input batches and samples - Dataloader
# start without batches, then add batching

TRAIN_FILE = 'data/train.npy'
TEST_FILE = 'data/test.npy'
NUM_FEATURES = 2
BATCH_SIZE = 64
SEQ_LENGTH = 100
NUM_EPOCHS = 10
MODEL_SAVE_FILE = 'model'
LOSS_SAVE_FILE = 'losses.npy'
SCORES_SAVE_FILE = 'scores.npy'

utils.checkGPU()

# define Datasets and Dataloaders
train_dataset = MIDIDataset(TRAIN_FILE)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = MIDIDataset(TEST_FILE)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=BATCH_SIZE)

# define model and hyperparameters
model = VelocityLSTM(NUM_FEATURES, F.relu)
model.cuda()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

def validate():
    print('Validating...')
    model.eval()
    losses = []
    for batch, targets in test_loader:
        try:
            model.zero_grad()
            output= model(batch.cuda())
            loss = loss_function(output,targets.cuda())
            losses.append(loss.item())
            pass
        except:
            e = sys.exc_info()[0]
            print(e)
            pass

    mean_loss = np.mean(losses)
    print(f'Loss: {round(mean_loss, 4)}')
    return mean_loss

def train():
    # training loop
    losses = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        print(f'Training epoch {epoch + 1}...')
        for batch, targets in train_loader:
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
    inputs, targets = next(iter(train_loader))
    scores = model(inputs.cuda())
    print(scores)
    np.save(SCORES_SAVE_FILE, scores.cpu())