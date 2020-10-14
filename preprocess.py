import numpy as np
from sklearn.model_selection import train_test_split
import time, argparse, utils

"""
Converts a NoteSequence into an array of note attributes
Args:
    seq (NoteSequence): The NoteSequence object to convert

Returns:
    (list(tuple)): A list of tuples with the following shape:
        (velocity, pitch, duration)
"""
def seq_to_arr(seq):
    notes = []
    for note in seq.notes:
        duration = round(note.end_time - note.start_time, 4)
        notes.append([note.velocity, note.pitch, duration])
    return notes

"""
Creates subsequences of N notes from the data
Args:
    seqs: (list(NoteSequence)) - An array of NoteSequence objects
    seq_length: (int) - The number of notes to include for each subsequence

Returns:

"""
def create_subseqs(seqs, seq_length):
    subseqs = []
    for seq in seqs:
        subseqs += utils.batch_data(seq, seq_length)
    return np.array(subseqs)

"""
Creates the test and train splits from the real velocities
"""
def create_test_train(batches, ratio=0.1):
    y = [batch[:,:,0] for batch in batches]
    x = [batch[:,:,1:] for batch in batches]
    
    return train_test_split(x, y, test_size=ratio)
 

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", dest = "input_file", type=str, help = "Input file containing NoteSequences")
parser.add_argument("--output_file", dest = "output_file", type=str, help = "Output file to write the train and test datasets to")
parser.add_argument('--seq_length', dest="seq_length", type=int, help="The number of notes per sample")
parser.add_argument('--batch_size', dest="batch_size", type=int, help="The number of samples per batch")
args = parser.parse_args()

start = time.time()
print(f'Loading {args.input_file} file...')
sequences = np.load(args.input_file, allow_pickle=True)
print(f'{len(sequences)} sequences loaded')
print(f'Creating samples of length {args.seq_length} notes...')
seq_arrs = [seq_to_arr(seq) for seq in sequences[:,1]]
samples = create_subseqs(seq_arrs, args.seq_length)
print(f'{len(samples)} samples created.')
print(f'Batching data in batches of {args.batch_size}...')
batches = utils.batch_data(samples, args.batch_size)
print(f'{len(batches)} batches created')
print('Creating training and testing sets')
datasets = create_test_train(batches)
print(f'Saving to {args.output_file}...')
np.save(args.output_file, datasets)
end = time.time() - start
print('Done')
print(f'Preprocessing took {round(end, 2)}s')
