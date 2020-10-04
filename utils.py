import ast
from note_seq.protobuf import music_pb2
from note_seq.performance_encoder_decoder import NotePerformanceEventSequenceEncoderDecoder

"""
Reads a dictionary file of NoteSequences. Note that depending
on the size of the file this may require several GBs of RAM

Args:
    file (string): The path to the notesequences file

Returns:
    dict(string, NoteSequence)
"""
def load_ns_file(file):
    with open(file, 'r') as file:
        filemaps = ast.literal_eval(file.read())
    for key in filemaps:
        filemaps[key] = music_pb2.NoteSequence().FromString(filemaps[key])
    return filemaps

def ns_to_tensor(ns):
    encoder = NotePerformanceEventSequenceEncoderDecoder(32)
    tensor = encoder.encode(ns)