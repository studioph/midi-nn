import torch, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VelocityLSTM
import numpy as np

utils.checkGPU()

train_mapping, test_mapping, x_train, x_test, y_train, y_test = np.load('data/train_test.npy', allow_pickle=True)

# convert data arrays to tensors
y_train = [torch.tensor(seq).long() for seq in y_train]
y_test = [torch.tensor(seq).long() for seq in y_test]

x_train = [utils.seq_to_tensor(seq) for seq in x_train]
x_test = [utils.seq_to_tensor(seq) for seq in x_test]

NUM_FEATURES = 3
SEQ_LENGTH = 100
NUM_EPOCHS = 100
SAVE_FILE = 'model'

model = VelocityLSTM(NUM_FEATURES)
model.cuda()
loss_function = nn.CrossEntropyLoss()
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
        for sequence, targets in zip(x_train, y_train):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Run our forward pass.
            scores = model(sequence.cuda())

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(scores, targets.cuda())
            loss.backward()
            optimizer.step()
        losses.append(validate())
    print('Training complete')
    return losses

losses = train()
torch.save(model, SAVE_FILE)

print('Final scores after training:')
with torch.no_grad():
    inputs = x_train[0].cuda()
    scores = model(inputs)
    print(scores)