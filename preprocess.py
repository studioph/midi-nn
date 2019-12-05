import numpy as np
from mido import MidiFile
from multiprocessing import Pool
from os import listdir

load_dir = './data/original/'
save_dir = './data/numpy/'

# Converts the MIDI file into an array of [index, pitch, delta, velocity] for each note_on event in the file
def midi_to_npy(file):
    print(file)
    mid = MidiFile(load_dir + file)
    track = mid.tracks[0]
    np.save(save_dir + file[:-4], [[idx, msg.note, msg.time, msg.velocity] for idx, msg in enumerate(track) if msg.type == 'note_on'])

if __name__ == '__main__':

    files = listdir(load_dir)

    p = Pool(12)
    p.map(midi_to_npy, files)