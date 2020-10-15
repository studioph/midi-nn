from note_seq import midi_file_to_note_sequence, sequence_proto_to_midi_file
import torch, os
import utils
import numpy as np
from note_seq.protobuf.music_pb2 import NoteSequence
"""
Functions to run one or more MIDI files through the LSTM model and integrate the output for playback
"""

def integrate_output(sequence: NoteSequence, results: torch.Tensor):
    for note, result, in zip(sequence.notes, results.tolist()):
        note.velocity = result

def sequence_midi_files(input_dir: str, output_dir: str, model_file: str, use_gpu=True):
    model = utils.load_model(model_file, use_gpu)
    files = [file for file in os.listdir(input_dir) if file.lower().endswith('.mid')]
    for file in files:
        print(f'Sequencing {file}...')
        notesequence = midi_file_to_note_sequence(file)
        seq_arr = utils.seq_to_arr(notesequence)
        inputs = np.array(seq_arr)[:,1:]
        if use_gpu:
            inputs.cuda()
        results = model(inputs)
        integrate_output(notesequence, results)
        sequence_proto_to_midi_file(notesequence, f'{output_dir}/{file}')
    print('Done')
