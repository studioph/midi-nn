import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

"""
Ensures GPU is detected

Raises:
    EnvironmentError - If the GPU cannot be detected by the ML library
"""
def checkGPU():
    if not torch.cuda.is_available():
        raise EnvironmentError('GPU not detected!')
        exit(1)

"""
Encodes the real velocities as a one-hot tensor of length 128
"""
def encode_velocities(arr: list):
    ohvs = []
    for idx, value in enumerate(arr):
        ohv = torch.zeros(128) # 128 velocities to choose from
        ohv[int(value)] = 1
        ohvs.append(ohv)
    return torch.cat(ohvs).view(len(ohvs), 1, -1).float()

"""
Converts array of target arrays into one-hot tensors
"""
def encode_dataset(ds):
    return [encode_velocities(arr) for arr in ds]

"""
Extracts batches of length N from an array of samples
Args:

Returns:
"""
def batch_data(arr: list, batch_size: int):
    batches = []
    idxs = range(len(arr) + batch_size)[0::batch_size]
    for idx in range(len(idxs[:-1])):
        batches.append(arr[idxs[idx]:idxs[idx + 1]])
    # discard incomplete batches
    if len(batches[-1]) != batch_size:
        batches.pop()
    return batches

"""
Loads a saved Pytorch model to either the CPU or GPU (default)

Returns:
    A Pytorch model bound to the GPU
"""
def load_model(model_file_path: str, use_gpu=True):
    model = torch.load(model_file_path)
    return model.cuda()

"""
Converts a NoteSequence into an array of note attributes
Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (list(tuple)): A list of tuples with the following shape:
        (velocity, pitch, duration)
"""
def seq_to_arr(seq: NoteSequence):
    notes = []
    for note in seq.notes:
        duration = round(note.end_time - note.start_time, 4)
        notes.append([note.velocity, note.pitch, duration])
    return notes


"""
Plots the losses during training and validation as a line graph using Matplotlib

Args:
    train_losses (numpy.ndarray): The training losses
    test_losses (numpy.ndarray): The validation losses
    num_epochs (int): The number of epochs the model was trained with
    lr (float): The learning rate the model was trained with
    batch_size (int): The batch size the model was trained with
"""
def plot_losses(train_losses: np.ndarray, test_losses: np.ndarray, num_epochs: int, lr: float, batch_size: int):
    y = [i + 1 for i in range(num_epochs)]
    fig, ax = plt.subplots()
    plt.plot(y, train_losses, label='Train loss')
    plt.plot(y, test_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train and validation losss with batch size {batch_size} and learning rate {lr}')
    plt.legend()
    plt.savefig(f'losses/{datetime.now().isoformat()}.png')
    plt.show()
