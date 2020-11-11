from note_seq import midi_file_to_note_sequence, sequence_proto_to_midi_file
import torch, os
from midi_nn import utils
import numpy as np
from note_seq.protobuf.music_pb2 import NoteSequence

"""
Functions to run one or more MIDI files through the LSTM model and integrate the output for playback
"""

def integrate_output(sequence: NoteSequence, results: list):
    for note, result, in zip(sequence.notes[1:], results):
        note.velocity = int(result)

def sequence_midi_files(input_dir: str, output_dir: str, model_file: str):
    model = utils.load_model(model_file)
    files = [file for file in os.listdir(input_dir) if file.lower().endswith('.mid')]
    for file in files:
        print(f'Sequencing {file}...')
        notesequence = midi_file_to_note_sequence(f'{input_dir}/{file}')
        seq_arr = utils.seq_to_arr(notesequence, 8)
        num_features = len(seq_arr[0][1:])
        inputs = torch.tensor(np.array(seq_arr)[:,1:]).view(1, -1, num_features)
        inputs = inputs.cuda().float()
        with torch.no_grad():
            results = model(inputs)
        integrate_output(notesequence, results.view(-1).tolist())
        sequence_proto_to_midi_file(notesequence, f'{output_dir}/{file}')

    print('Done')


input_dir = 'data/inputs'
output_dir = 'data/results'
model_file = 'model.zip'
sequence_midi_files(input_dir, output_dir, model_file)