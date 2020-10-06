import numpy as np
from sklearn.model_selection import train_test_split
import time

"""
Converts a NoteSequence into an array of shape 
    (seq_length, (velocity, pitch, duration))
Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (torch.tensor): A tensor with the following shape:
        (# of notes, 3)
"""
def seq_to_arr(seq, seq_length):
    notes = []
    for note in seq.notes[:seq_length]:
        duration = round(note.end_time - note.start_time, 4)
        notes.append([note.velocity, note.pitch, duration])
    return notes

"""
Converts an array of NoteSequences

Args:
    seqs ([(name, NoteSequence)]): An array of NoteSequence objects

Returns: 
    (np.array): A tensor with the following shape:
        (len(seqs), (# of notes in sequence, 3))
"""
def seqs_to_arrs(seqs, seq_length):
    seq_arrs = []
    for seq in seqs:
        seq_arrs.append([seq[0], np.array(seq_to_arr(seq[1], seq_length))])
    return np.array(seq_arrs)

"""
Strips the actual note velocities from a NoteSequence, replacing
them with `value`.

Args:
    ns (NoteSequence): The sequence to strip velocities from
    value (int): The value to replace each note velocity with
        from 0-127. This function modifies the sequence in-place

"""
def set_note_velocities(seq, value=0):
    z = np.full(len(seq), value)
    seq[:,0] = z

"""
Strips the velocities of multiple sequences, setting the velocity
of each note to `value`
"""
def set_velocities(seqs, value=0):
    for seq in seqs:
        set_note_velocities(seq, value) 

"""
Creates a mapping of indexes to sequence names for reconstruction
"""
def map_names_to_indexes(tensors):
    idx_to_name = {}
    for idx, tensor in enumerate(tensors):
        idx_to_name[idx] = tensor[0]
    return idx_to_name

"""
Creates the test and train splits from the ground truths
"""
def create_test_train(seqs, seq_length, ratio=0.1):
    seq_arrs = seqs_to_arrs(seqs, seq_length)
    y = [np.copy(seq[:,0]) for seq in seq_arrs[:,1]]
    set_velocities(seq_arrs[:,1])
    x = seq_arrs
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio)
    train_mapping = map_names_to_indexes(x_train)
    test_mapping = map_names_to_indexes(x_test)
    x_train = x_train[:,1]
    x_test = x_test[:,1]
    return train_mapping, test_mapping, x_train, x_test, y_train, y_test

input_file = 'data/notesequences.npy'
output_file = 'data/train_test.npy'

start = time.time()
print(f'Loading {input_file} file...')
sequences = np.load(input_file, allow_pickle=True)
print('Creating training and testing sets')
datasets = create_test_train(sequences, seq_length=100)
print(f'Saving to {output_file}...')
np.save(output_file, datasets)
end = time.time() - start
print('Done')
print(f'Preprocessing took {round(end, 2)}s')
