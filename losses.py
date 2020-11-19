import numpy as np
from numpy.lib.function_base import median
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
print('acquiring losses from models...')
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

# plot losses as scatter plot
print('plotting losses...')
fig, ax = plt.subplots()
x_range = range(len(model_losses))
ax.scatter(x=['NN Model' for i in x_range], y=model_losses, color="g", marker=".", label="NN model loss")
ax.scatter(x=['Basic Predictor' for i in x_range], y=basic_losses, color="r", marker=".", label='Basic predictor loss')
ax.scatter(x=['Avg Predictor' for i in x_range], y=basic_avg_losses, color="b", marker=".", label="Avg predictor loss")
ax.set_xlabel('Model')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True)
ax.set_title('Losses between various predictor models predicting velocity')
plt.savefig('losses/loss_compare_scatter.png')
plt.show()

#plot median loss of each model as bar chart
median_losses = [np.median(model_losses), np.median(basic_losses), np.median(basic_avg_losses)]
fig, ax = plt.subplots()
plt.bar(x=['NN Model', 'Basic Predictor', 'Avg Predictor'], height=median_losses)
plt.xlabel('Model')
plt.ylabel('Median loss')
plt.title('Median loss of predictor models predicting velocity')
plt.savefig('losses/loss_compare_bar.png')
plt.show()