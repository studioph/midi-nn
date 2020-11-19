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

def sequence_note(note_vec, model, prev):
    note_vec[4] = prev
    inputs = torch.tensor(note_vec[1:]).view(1,1,-1).float()
    with torch.no_grad():
        pred_vel = model(inputs.cuda())
        return pred_vel.item()

def sequence_midi_file(seq_arr, model, num_features, n_steps):
    # sequence notes with velocity from MIDI file
    inputs = torch.tensor(np.array(seq_arr[:n_steps])[:,1:]).view(1,-1,num_features).float()
    with torch.no_grad():
        pred_vels = model(inputs.cuda())
        results = pred_vels.view(-1).tolist()
    # sequence the rest of the notes with the predicted velocities
    for note_vec in seq_arr[n_steps:]:
        prev_vel = results[-1]
        results.append(sequence_note(note_vec, model, prev_vel))
    return results

def sequence_midi_files_generative(input_dir: str, output_dir: str, model_file: str):
    model = utils.load_model(model_file)
    files = [file for file in os.listdir(input_dir) if file.lower().endswith('.mid')]
    for file in files:
        print(f'Sequencing {file}...')
        notesequence = midi_file_to_note_sequence(f'{input_dir}/{file}')
        seq_arr = utils.seq_to_arr(notesequence, 8)
        num_features = len(seq_arr[0][1:])
        results = sequence_midi_file(seq_arr, model, num_features, 10)
        integrate_output(notesequence, results)
        sequence_proto_to_midi_file(notesequence, f'{output_dir}/{file[:-4]}_gen.mid')

    print('Done')

def sequence_midi_files_transform(input_dir: str, output_dir: str, model_file: str):
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
        sequence_proto_to_midi_file(notesequence, f'{output_dir}/{file[:-4]}_trans.mid')


input_dir = 'data/inputs'
output_dir = 'data/results'
model_file = 'model_40.zip'
sequence_midi_files_generative(input_dir, output_dir, model_file)