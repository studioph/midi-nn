from note_seq import midi_file_to_note_sequence
import json, argparse, os, time
from multiprocessing import Pool
import numpy as np

"""
Converts a list of MIDI files into a dictionary map of filenames
to bytes-serialized NoteSequences

Args:
    args (string, list<string>): a tuple containing the input
        directory path and a list of MIDI file names

Returns:
    [filename, NoteSequence (bytes)]
"""
def convert_midi_files(args):
    input_dir, files = args
    filemap = []
    for file in files:
        print(f'Converting {file}...')
        filename = file[:-4]
        input_path = input_dir + '/' + file
        sequence = midi_file_to_note_sequence(input_path)
        filemap.append([filename, sequence])
    return filemap

"""
Merges arrays

Args:
    [filename, NoteSequence]

Returns:
    A single level array with filename, NoteSequence pairs
"""
def merge(arrs):
    merged = []
    for arr in arrs:
        merged += arr
    return merged

"""
Splits an iterable (array, list) into n sublists

Args:
    arr (iterable): An iterable object
    n (int): The number of pieces to split into

Returns:
    A list of length n containing equally sized sublists of arr
"""
def split(arr, n):
    k, m = divmod(len(arr), n)
    return (arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --input_dir FOLDER --output_file FOLDER --threads 4
    parser.add_argument("--input_dir", dest = "input_dir", help = "Input folder containing MIDI files")
    parser.add_argument("--output_file", dest = "output_file", help = "Output file to write converted NoteSequences to")
    parser.add_argument("--processes", dest="processes", type=int, default=4, help="Number of processes to use")

    args = parser.parse_args()
    start = time.time()

    files = [file for file in os.listdir(args.input_dir) if file.lower().endswith('.mid')]
    split_files = list(split(files, args.processes))

    # parallelize the conversions. Merge the dictionaries at the end
    with Pool(args.processes) as pool:
        results = pool.map(convert_midi_files, [(args.input_dir, split_files[i]) for i in range(len(split_files))])
    filemaps = merge(results)
    end = time.time() - start

    # Write dictionary to file for later use
    np.save(args.output_file, filemaps, allow_pickle=False)
    print('Done')
    print(f'Conversion took {round(end, 2)}s')