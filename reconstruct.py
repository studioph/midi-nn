from mido import MidiFile
import numpy as np 

load_dir = './data/original/'
save_dir = './data/results/'

# merges the results from the NN with the indecies from the file
def merge_idxs(idxs, results):
    return [[[idxs[i][j], results[i][j]] for j in range(len(idxs[0]))] for i in range(len(idxs))]
    # data should now be in the shape of (batch_size, (seq_len, 2)) with the 2 being [index, velocity]