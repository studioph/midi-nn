import ast
from note_seq.protobuf import music_pb2
from note_seq.performance_encoder_decoder import NotePerformanceEventSequenceEncoderDecoder
import numpy as np

def ns_to_tensor(seqs):
    encoder = NotePerformanceEventSequenceEncoderDecoder(32)
    tensor = encoder.encode([seq.notes for seq in seqs])
    return tensor