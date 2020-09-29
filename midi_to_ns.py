from note_seq import midi_file_to_note_sequence
import json, argparse, os

parser = argparse.ArgumentParser()

# --input_dir FOLDER --output_file FOLDER --threads 4
parser.add_argument("--input_dir", dest = "input_dir", help = "Input folder containing MIDI files")
parser.add_argument("--output_file", dest = "output_file", help = "Output file to write converted NoteSequences to")
# parser.add_argument("--threads", dest="threads", type=int, default=4, help="Number of threads to use")

args = parser.parse_args()

files = [file for file in os.listdir(args.input_dir) if file.lower().endswith('.mid')]
filemap = {} # dictionary of filenames to NoteSequences for reconstruction

for file in files:
    print(f'Converting {file}...')
    filename = file[:-4]
    input_path = args.input_dir + '/' + file
    sequence = midi_file_to_note_sequence(input_path)
    filemap[filename] = sequence.SerializeToString()

# Write dictionary to file for later use
with open(args.output_file, 'w') as outfile:
    outfile.write(str(filemap))
print('Done')