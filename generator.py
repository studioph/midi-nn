import numpy as np

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
    def __init__(self, x_data, y_data, num_steps=1, batch_size=1):
        self.x_data = x_data
        self.y_data = y_data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.current_idx = 0

    def generate(self):
        while True:
            y = np.zeros(())
            if self.current_idx >= len(self.x_data):
                self.current_idx = 0 # reset back to start
            x = self.x_data[self.current_idx][1] # get the tensor
            # create one-hot vector for real velocity value
            one_hot = np.zeros(128) # 128 velocities to choose from
            one_hot
            yield x, y
