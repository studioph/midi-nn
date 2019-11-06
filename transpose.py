from mido import MidiFile
import os
import random

data_dir = './data/yamaha_epiano'
files = os.listdir(data_dir)

def transpose(msg, step, up):
    #transpose up or down
    if up: msg.note = msg.note + step               
    else: msg.note = msg.note - step 

for file in files:
    try:
        print('Procesing ' + file)
        #create random number to transpose track by 1 (half step) to 4 (major 3rd)
        step = random.randint(1, 4)
        #determines whether to transpose up or down
        up = bool(random.getrandbits(1))

        #load MIDI file
        path = data_dir + '/' + file
        mid = MidiFile(path)
        track = mid.tracks[0]
        #only change note on/off events
        [transpose(msg, step, up) for msg in track if msg.type == 'note_on' or msg.type == 'note_off']      
                
        #create new file name
        if up: path = path[:-4] + '_transpose_up_' + str(step) + '.mid'
        else: path = path[:-4] + '_transpose_down_' + str(step) + '.mid'
        mid.save(path)
    except:
        print('Error with ' + file)

