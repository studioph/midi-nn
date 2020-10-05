import utils, subprocess
import numpy as np

# subprocess.run([
#     'python3', 'midi_to_ns',
#     '--input_dir', 'data/midi',
#     '--output_file', 'data/notesequences.npy',
#     '--processes', '16'
#     ],capture_output=True)

sequences = np.load('data/notesequences.npy', allow_pickle=True)
train_mapping, test_mapping, x_train, x_test, y_train, y_test = utils.create_test_train(sequences, seq_length=100)
