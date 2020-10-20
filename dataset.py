import torch
import numpy as np

class MIDIDataset(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.samples = np.load(data_file, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.samples[index][:,1:]
        y = self.samples[index][:,0]
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        return x, y