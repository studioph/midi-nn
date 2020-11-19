import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import loss
from midi_nn.basic import BasicVelocityPredictor
import matplotlib.pyplot as plt
from midi_nn import utils

test_file = 'data/test.npy'
model_file = 'model_40.zip'
loss_function = nn.MSELoss()
predictor = BasicVelocityPredictor()

model_losses = []
basic_losses = []
basic_avg_losses = []

# using RMSE since it makes more logical sense in context of velocities
def RMSELoss(yhat, y):
    return torch.sqrt(loss_function(yhat, y))

test = np.load(test_file)
model = utils.load_model(model_file)

# get losses for each validation sample
for sample in test:
    targets = torch.tensor(sample[:,0]).float()
    input = torch.tensor(sample[:,1:]).float().view(1, -1, 7)

    #get model losses
    model_output = model(input.cuda())
    model_loss = RMSELoss(model_output.view(-1), targets.cuda())
    model_losses.append(model_loss.item())

    #get basic predictor losses
    basic_output = predictor.predict(targets)
    basic_loss = RMSELoss(basic_output, targets)
    basic_losses.append(basic_loss.item())

    #get basic avg predictor losses
    basic_avg_output = predictor.predict_avg(targets)
    basic_avg_loss = RMSELoss(basic_avg_output, targets)
    basic_avg_losses.append(basic_avg_loss.item())

# plot losses
fig, ax = plt.subplots()
y = [i + 1 for i in range(len(test))]
plt.xlabel('Validation Sample')
plt.ylabel('Loss')
plt.plot(y, model_losses, label="Model losses")
plt.plot(y, basic_losses, label="Basic Predictor losses")
plt.plot(y, basic_avg_losses, label="Basic Average Predictor losses")
plt.legend()
plt.title('Losses between various predictor models predicting velocity')
plt.show()