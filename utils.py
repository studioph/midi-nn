import numpy as np
import torch

"""
Ensures GPU is detected
"""
def checkGPU():
    if not torch.cuda.is_available():
        raise EnvironmentError('GPU not detected!')
        exit(1)

"""
Encodes the real velocities as a one-hot tensor of length 128
"""
def encode_velocities(arr):
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
def batch_data(arr, batch_size):
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
"""
def load_model(model_file_path, use_gpu=True):
    model = torch.load(model_file_path)
    if use_gpu:
        model.cuda()
    return model

"""
Converts a NoteSequence into an array of note attributes
Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (list(tuple)): A list of tuples with the following shape:
        (velocity, pitch, duration)
"""
def seq_to_arr(seq):
    notes = []
    for note in seq.notes:
        duration = round(note.end_time - note.start_time, 4)
        notes.append([note.velocity, note.pitch, duration])
    return notes