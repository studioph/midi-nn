from note_seq import midi_file_to_note_sequence, midi_file_to_sequence_proto
from note_seq.protobuf import music_pb2
ns = midi_file_to_note_sequence('data/test4.mid')
sequence = music_pb2.NoteSequence()
print(ns)