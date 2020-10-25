import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from note_seq import midi_file_to_note_sequence

##################################################################
# A collection of functions used by the different project scripts
##################################################################

##########
# GENERAL
##########

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
Loads a saved Pytorch model to either the CPU or GPU (default)

Returns:
    A Pytorch model bound to the GPU
"""
def load_model(model_file_path: str, use_gpu=True):
    model = torch.load(model_file_path)
    return model.cuda()


##################
# MIDI CONVERSION
##################

"""
Converts a list of MIDI files into a list of NoteSequences

Args:
    args (string, iterable<string>): a tuple containing the input
        directory path and an of MIDI file names

Returns:
    list(NoteSequence) - A list NoteSequence objects
"""
def convert_midi_files(args: tuple):
    input_dir, files = args
    sequences = []
    for file in files:
        print(f'Converting {file}...')
        filename = file[:-4]
        input_path = input_dir + '/' + file
        sequence = midi_file_to_note_sequence(input_path)
        sequences.append(sequence)
    return sequences

#####################
# DATA PREPROCESSING
#####################

"""
Extracts batches of length N from an array of samples
Args:
    arr (iterable) - An array-like object to be split into batches
    batch_size (int) - The size of each batch
Returns:
"""
def batch_data(arr: list, batch_size: int):
    leftover = len(arr) % batch_size # discard excess samples
    if leftover > 0:
        arr = arr[:-leftover]
    subarr = np.array(arr)
    batches = np.split(subarr, int(len(arr) / batch_size))
    return batches

"""
Creates subsequences of N notes from full NoteSequences
Args:
    seqs: (list(NoteSequence)) - An array of NoteSequence objects
    seq_length: (int) - The number of notes to include for each subsequence

Returns:
    numpy.ndarray - A Numpy array containing subsequences of length N
"""
def create_subseqs(seqs: list, seq_length: int):
    subseqs = []
    for seq in seqs:
        subseqs += batch_data(seq, seq_length)
    return np.array(subseqs)

"""
Converts a NoteSequence into an array of note attributes. 
Note: starts with the second note of the sequence.

Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (list(tuple)): A list of tuples with the following shape:
        (velocity, pitch, duration, prev note velocity)
"""
def seq_to_arr(seq):
    notes = []
    for idx, note in enumerate(seq.notes[1:], start=1):
        duration = round(note.end_time - note.start_time, 4)
        notes.append([
            note.velocity, 
            note.pitch, 
            duration,
            seq.notes[idx - 1].velocity])
    return notes

##########################
# TRAINING AND VALIDATION
##########################
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
