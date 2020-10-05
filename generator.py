import numpy as np
from utils import encode_velocities
"""
Takes in the XY NoteSequence data and creates a batch for the model

Args:
    x_data([(name, NoteSequence)]): The NoteSequences with stripped velocities
    y_data([(name, NoteSequence)]): The original NoteSequences
    batch_size (int): The number of sequences or partial sequences to feed

Yields:
    x,y ()
"""
class KerasBatchGenerator(object):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.current_idx = 0

    def generate(self):
        while True:
            if self.current_idx >= len(self.x_data):
                self.current_idx = 0 # reset back to start
            x = self.x_data[self.current_idx] # get the tensor
            # encode one-hot tensor of actual velocities
            y = np.array(encode_velocities(tensor) for tensor in self.y_data)
            yield x, y
