from mido import MidiFile
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import random

def get_attributes(mid):
    track = mid.tracks[0] #assuming single track MIDI file in this scenario, as all of the ones in the dataset are
    #filter for just the note_on messages
    note_on_msgs = [msg for msg in track if msg.type == 'note_on']
    #convert the message into a matrix of [pitch, time]
    data = [[msg.note, msg.time] for msg in note_on_msgs]
    return data
def get_velocities(mid):
    track = mid.tracks[0] #assuming single track MIDI file in this scenario, as all of the ones in the dataset are
    #filter for just the note_on messages
    note_on_msgs = [msg for msg in track if msg.type == 'note_on']
    data = [msg.velocity for msg in note_on_msgs]
    return data

def get_files(path='.', size=None, filter=True):
    if filter:
        files = [file for file in os.listdir(path) if 'transpose' not in file][:size]
    else:
        files = [file for file in os.listdir(path)][:size]
    return files

def load_midi(path, file):
   
    # load the files using mido
    # keep the filename attached to each matrix for reintegration
    pair = [file, MidiFile(path + file)]
    x = [[pair[0], get_attributes(pair[1])]
    y = [[pair[0], get_velocities(pair[1])]
    return x, y

def split(midis, size):
    x, y = midis
    new_x, new_y = [], []

    # split x into equal size chunks
    for pair in x:
        idxs = [i * size for i in range(int(len(pair[1]) / size))]
        for idx in idxs:
            # keep the filename tagged along with the index in the original array where this slice came from
            new_x.append([pair[0], idx, pair[1][idx:idx + size]])

    # split x into equal size chunks
    for pair in y:
        idxs = [i * size for i in range(int(len(pair[1]) / size))]
        for idx in idxs:
            # keep the filename tagged along with the index in the original array where this slice came from
            new_y.append([pair[0], idx, pair[1][idx:idx + size]])

    return new_x, new_y

#splits and saves the data into numpy files
def split_and_save(data, path, test_size=0.2):
    x_train, x_test, y_train, y_test = train_test_split(data[0], data[1], test_size=test_size)
    np.save(path + 'x_train', x_train)
    np.save(path + 'x_test', x_test)
    np.save(path + 'y_train', y_train)
    np.save(path + 'y_test', y_test)

def load_data(path):
    x_train = np.load(path + 'x_train')
    x_test = np.load(path + 'x_test')
    y_train = np.load(path + 'y_train')
    y_test = np.load(path + 'y_test')

    return x_train, x_test, y_train, y_test

def reconstruct_files(metadata, yhat, savepath, loadpath):
    # choose a random batch to reconstruct MIDI files from
    idx = random.randint(0, len(yhat))
    metadata = metadata[idx]
    yhat = yhat[idx]
    for i, sequence in enumerate(metadata):
        reconstruct(sequence, yhat[i], savepath, loadpath)

#reconstructs a single MIDI file
def reconstruct(sequence, yhat, savepath, loadpath):
    filename = sequence[0]
    idx = sequence[1]
    mid = MidiFile(loadpath + filename)
    track = mid.tracks[0]
    notes = [msg for msg in track if msg.type == 'note_on']
    notes = notes[idx:idx + len(yhat)]
    for i in range(yhat):
        notes[i].velocity = int(yhat[i])

    mid.save(savepath + filename)

# def batch(data, batch_size):


data_dir = './data/original/'

data = load_midi(path=data_dir, size=100)
data = split(data, 100)
# data = batch(data, 10)
split_and_save(data, './data/numpy/')