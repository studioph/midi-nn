import ast
from note_seq.protobuf import music_pb2
from note_seq.performance_encoder_decoder import NotePerformanceEventSequenceEncoderDecoder
import numpy as np

"""
Converts a NoteSequence into a tensor of shape 
    (sequence_name, len(seq.notes), (velocity, pitch, duration))
Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (np.array): A tensor with the following shape:
        (2, (# of notes, 3))
"""
def ns_to_tensor(seq):
    tensor = []
    for note in seq[1].notes:
        duration = round(note.end_time - note.start_time, 4)
        tensor.append([note.velocity, note.pitch, duration])
    return np.array([seq[0], tensor])

"""
Converts an array of NoteSequences into tensors

Args:
    seqs ([NoteSequence]): An array of NoteSequence objects

Returns: 
    (np.array): A tensor with the following shape:
        (len(seqs), (2, (# of notes in sequence, 3)))
"""
def ns_to_tensors(seqs):
    tensor = []
    for seq in seqs:
        tensor.append(ns_to_tensor(seq))
    return np.array(tensor)