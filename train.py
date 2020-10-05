import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import VelocityLSTM
from utils import encode_velocities
import numpy as np

train_mapping, test_mapping, x_train, x_test, y_train, y_test = np.load('data/train_test.npy', allow_pickle=True)

# create one-hot tensors from actual velocities
y_train = np.array([encode_velocities(tensor) for tensor in y_train])
y_test = np.array([encode_velocities(tensor) for tensor in y_test])

NUM_FEATURES = 3
SEQ_LENGTH = 100

model = VelocityLSTM(NUM_FEATURES)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# training loop
for epoch in range(10):
    print(f'Training epoch {epoch + 1}...')
    for sequence, targets in zip(x_train, y_train):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Run our forward pass.
        scores = model(sequence)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(scores, targets)
        print(f'Loss: {loss}')
        loss.backward()
        optimizer.step()
        
print('Training complete')
# See what the scores are after training
with torch.no_grad():
    inputs = (x_train[0], y_train, [0])
    scores = model(inputs)
    print(scores)