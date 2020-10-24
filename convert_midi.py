import json, argparse, os, time
from multiprocessing import Pool
import numpy as np
from midi_nn import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # --input_dir FOLDER --output_file FILE --processes 4
    parser.add_argument("--input_dir", dest = "input_dir", help = "Input folder containing MIDI files")
    parser.add_argument("--output_file", dest = "output_file", help = "Output file to write converted NoteSequences to")
    parser.add_argument("--processes", dest="processes", type=int, default=4, help="Number of processes to use in parallel")

    args = parser.parse_args()
    start = time.time()

    files = [file for file in os.listdir(args.input_dir) if file.lower().endswith('.mid')]
    split_files = np.array_split(files, args.processes)

    # Parallelize the conversions. Merge the arrays at the end
    with Pool(args.processes) as pool:
        results = pool.map(utils.convert_midi_files, [(args.input_dir, split_files[i]) for i in range(args.processes)])
    sequences = np.concatenate(results)

    # Write sequences to file for later use
    np.save(args.output_file, sequences)
    end = time.time() - start
    print('Done')
    print(f'Conversion took {round(end / 60, 1)}m')