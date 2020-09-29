import ast
from note_seq.protobuf import music_pb2

with open('path to file', 'r') as file:
    dictionary = ast.literal_eval(file.read())

sequence = music_pb2.NoteSequence().FromString(dictionary['key'])