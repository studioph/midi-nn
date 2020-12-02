import numpy as np
from numpy.lib.function_base import median
import torch, random
import torch.nn as nn
from torch.nn.modules import loss
from midi_nn.basic import BasicVelocityPredictor
import matplotlib.pyplot as plt
from midi_nn import utils

test_file = 'data/test.npy'
model_file = 'model_40.zip'
loss_function = nn.MSELoss()

model_losses = []
basic_losses = []
basic_avg_losses = []
whole_avg_losses = []

# using RMSE since it makes more logical sense in context of velocities
def RMSELoss(yhat, y):
    return torch.sqrt(loss_function(yhat, y))

test = np.load(test_file)
model = utils.load_model(model_file)

# get avg velocity for the whole test dataset
velocities = test[:,:,0]
shape = np.shape(velocities)
velocities = np.reshape(velocities, shape[0] * shape[1])
avg = round(np.mean(velocities), 4)

predictor = BasicVelocityPredictor(avg)

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

    #get basic avg predictor losses for sample
    basic_avg_output = predictor.predict_sample_avg(targets)
    basic_avg_loss = RMSELoss(basic_avg_output, targets)
    basic_avg_losses.append(basic_avg_loss.item())

    # get basic avg predictor losses for whole dataset
    whole_avg_output = predictor.predict_whole_avg(targets)
    whole_avg_loss = RMSELoss(whole_avg_output, targets)
    whole_avg_losses.append(whole_avg_loss.item())

# plot losses as histogram for each predictor
print('plotting losses...')
fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
fig.suptitle('Distribution of losses among model types')
for pred, losses, x, y in zip(['NN Model', 'Basic Predictor', 'Sample Avg', 'Dataset Avg'], 
                            [model_losses, basic_losses, basic_avg_losses, whole_avg_losses],
                            [0,0,1,1], [0,1,0,1]):
    axs[x][y].hist(losses)
    axs[x][y].set_title(pred)
    axs[x][y].set_xlabel('Loss')
plt.show()

#plot median loss of each model as bar chart
median_losses = [np.median(model_losses), np.median(basic_losses), np.median(basic_avg_losses), np.median(whole_avg_losses)]
fig, ax = plt.subplots()
plt.bar(x=['NN Model', 'Basic Predictor', 'Sample Avg', 'Dataset Avg'], height=median_losses)
plt.xlabel('Model')
plt.ylabel('Median loss')
plt.title('Median loss of predictor models predicting velocity')
plt.show()

# plot time series of losses for a single sample
sample = random.choice(test)
targets = torch.tensor(sample[:,0]).float()
input = torch.tensor(sample[:,1:]).float().view(1, -1, 7)

actual_vel = sample[:,0]
model_vel = model(input.cuda()).view(-1).tolist()
basic_vel = predictor.predict(targets).tolist()
sample_avg = predictor.predict_sample_avg(targets).tolist()
whole_avg = predictor.predict_whole_avg(targets).tolist()

fig, ax = plt.subplots()
plt.plot(actual_vel, color='royalblue', label='Actual')
plt.plot(model_vel, color='green', label='NN Model')
plt.plot(basic_vel, color='orange', label='Basic')
plt.plot(sample_avg, color='indianred', label='Sample Avg')
plt.plot(whole_avg, color='purple', label='Whole avg')
plt.title('Time series comparison of model losses')
plt.legend()
plt.ylabel('Velocity')
plt.xlabel('Note')
plt.show()