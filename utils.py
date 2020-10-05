from note_seq.protobuf import music_pb2
from note_seq.performance_encoder_decoder import NotePerformanceEventSequenceEncoderDecoder
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.test import is_gpu_available

# quit and print an error if there's no GPU
def checkGPU():
    if not is_gpu_available():
        raise EnvironmentError("No GPU detected!")
        exit(code=1)

"""
Converts a NoteSequence into a tensor of shape 
    (len(seq.notes), (velocity, pitch, duration))
Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (np.array): A tensor with the following shape:
        (# of notes, 3)
"""
def ns_to_tensor(seq):
    tensor = []
    for note in seq.notes:
        duration = round(note.end_time - note.start_time, 4)
        tensor.append([note.velocity, note.pitch, duration])
    return np.array(tensor)

"""
Converts an array of NoteSequences into tensors

Args:
    seqs ([(name, NoteSequence)]): An array of NoteSequence objects

Returns: 
    (np.array): A tensor with the following shape:
        (len(seqs), (# of notes in sequence, 3))
"""
def ns_to_tensors(seqs):
    tensors = []
    for seq in seqs:
        tensors.append([seq[0], ns_to_tensor(seq[1])])
    return np.array(tensors)

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
def create_test_train(seqs, ratio=0.1, saveFile=True):
    tensors = ns_to_tensors(seqs)
    y = np.array([np.copy(tensor[:,0]) for tensor in tensors[:,1]])
    set_velocities(tensors[:,1])
    x = tensors
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ratio)
    train_mapping = map_names_to_indexes(x_train)
    test_mapping = map_names_to_indexes(x_test)
    x_train = x_train[:,1]
    x_test = x_test[:,1]
    if saveFile:
        np.save('data/train_test', [train_mapping, test_mapping, x_train, x_test, y_train, y_test])
    return train_mapping, test_mapping, x_train, x_test, y_train, y_test

"""
Encodes the real velocities as a one-hot tensor of length 128
"""
def encode_velocities(tensor):
    z = []
    for idx, value in enumerate(tensor):
        ohv = np.zeros(128) # 128 velocities to choose from
        ohv[int(value)] = 1
        z.append(ohv)
    return np.array(z)