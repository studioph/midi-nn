import torch
import numpy as np

"""
Dataset wrapper for use with Pytorch DataLoaders.

Reads in a file containing NoteSequence subsequences, and creates x,y train/test sample pair
    for each subsequence in the file.
"""
class MIDIDataset(torch.utils.data.Dataset):
    def __init__(self, data_file: str):
        self.samples = np.load(data_file, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index][:,1:]
        y = self.samples[index][:,0] # 1st col is velocity
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        return x, y