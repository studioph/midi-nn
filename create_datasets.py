import numpy as np
import time, argparse
from sklearn.model_selection import train_test_split
from midi_nn import utils

# TODO
# things to add to model input:
# time delta of previous note
# time delta of next note
# velocity of previous note
# velocity of next note?
# articulation? Investigate Dorico files

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", dest = "input_file", type=str, help = "Input file containing NoteSequences")
parser.add_argument("--output_dir", dest = "output_dir", type=str, help = "Output directory to write the train and test datasets to")
parser.add_argument('--seq_length', dest="seq_length", type=int, help="The number of notes per sample")
parser.add_argument('--test_size', dest="test_size", type=float, default=0.1, help="The ratio of train/test samples")
args = parser.parse_args()

start = time.time()

print(f'Loading {args.input_file} file...')
sequences = np.load(args.input_file, allow_pickle=True)
print(f'{len(sequences)} sequences loaded')

print(f'Creating samples of length {args.seq_length} notes...')
seq_arrs = [utils.seq_to_arr(seq) for seq in sequences]
samples = utils.create_subseqs(seq_arrs, args.seq_length)

print(f'Creating train and test splits...')
train, test = train_test_split(samples, test_size=args.test_size)

print(f'Saving to {args.output_dir}/train.npy and {args.output_dir}/test.npy...')
np.save(f'{args.output_dir}/train.npy', train)
np.save(f'{args.output_dir}/test.npy', test)

end = time.time() - start

print('Done')
print(f'Preprocessing took {round(end, 2)}s')
