from mido import MidiFile, MidiTrack
import numpy as np 
from os import listdir
from copy import deepcopy


def load_data(load_dir, size_limit=None):
    files = [file for file in listdir(load_dir) if 'transpose' not in file]
    return [(file[:-4], np.load(load_dir + file)) for file in files][:size_limit]


def sequence_data(data, seq_len):
    sequences = []
    for arr in data:
        num_sequences = int(len(arr[1]) / seq_len)
        idxs = [x * seq_len for x in range(num_sequences)]
        # (filename, <sequence>)
        sequences += [np.array([arr[0], np.array(arr[1][i:i + seq_len])]) for i in idxs]
    return np.array(sequences)

def batch_data(sequences, batch_size):
    num_batches = int(len(sequences) / batch_size)
    idxs = [x * batch_size for x in range(num_batches)]
    batches = [np.array(sequences[i:i + batch_size]) for i in idxs]
    # batches += np.array([sequences[idxs[-1] + batch_size:]]) #last batch will not be full size       
    return batches

def split(data, train_size):
    idx = int(len(data) * train_size)
    train = data[:idx]
    test = data[idx:]
    return train, test

def reconstruct(batch, midi_dir, save_dir):
    filenames, idxs, velocities = batch
    velocities = velocities.astype(int)
    for i, filename in enumerate(filenames):
        print('reconstructing file ' + filename + '_' + str(i) + '.mid')
        mid = MidiFile(midi_dir + filename + '.mid')
        track = mid.tracks[0]
        #save original chunk
        orig_mid = deepcopy(mid)
        split_file(orig_mid, track, int(idxs[i][0]), int(idxs[i][-1]))
        orig_mid.save(save_dir + filename + '_' + str(i) + '_orig.mid',)
        for j, idx in enumerate(idxs[i]):
            track[int(idx)].velocity = velocities[i][j]
        split_file(mid, track, int(idxs[i][0]), int(idxs[i][-1]))
        mid.save(save_dir + filename + '_' + str(i) + '_res.mid',)
    print('done');

def split_file(mid, track, start_idx, end_idx):
    meta_msgs = [msg for msg in track if msg.is_meta][:-1]
    track = track[start_idx:end_idx]
    track = meta_msgs + track
    mid.tracks[0] = MidiTrack()
    for msg in track:
        mid.tracks[0].append(msg)

# merges the results from the NN with the indecies from the file
def merge_idxs(idxs, results):
    return [[[idxs[i][j], results[i][j]] for j in range(len(idxs[0]))] for i in range(len(idxs))]
    # data should now be in the shape of (batch_size, (seq_len, [index, velocity]))

