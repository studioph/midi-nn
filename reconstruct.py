from mido import MidiFile
import numpy as np 
from random import randint
import os

#randomly select a sequence of notes
results = np.load('res.npy')
n = randint(0, results.shape[0])
group = results[n]
r = randint(0, group.shape[0])
seq = group[r]

data_dir = './data/yamaha_epiano/'
files = os.listdir(data_dir)
a = randint(0, len(files))
mid = MidiFile(data_dir + files[a])
track = mid.tracks[0]
notes = [msg for msg in track if msg.type == 'note_on']
for idx in range(len(seq)):
    notes[idx].velocity = int(seq[idx])

mid.save('result.mid')