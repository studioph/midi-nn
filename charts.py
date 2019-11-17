import numpy as np 
import matplotlib.pyplot as plt 
from os import listdir
import csv

data_dir = './data/numpy/'
save_dir = './data/charts/'

files = [file for file in listdir(data_dir) if 'transpose' not in file] #transposed files should not be included in data analysis

data = [np.load(data_dir + file) for file in files]

pitches = [arr[:,1] for arr in data]
pitches_concat = np.concatenate(pitches)
velocities = [arr[:,3] for arr in data]
avg_velocities = [int(np.average(arr)) for arr in velocities]
velocities_concat = np.concatenate(velocities)
lengths = [len(arr) for arr in data]

# scatter plot
plt.scatter(lengths, avg_velocities)
plt.xlabel('Length')
plt.ylabel('Avg Velocity')
plt.title('Length vs Avg Velocity')
plt.savefig(save_dir + 'scatter.png')
plt.close()

# MIDI file length plot
plt.hist(lengths, rwidth=0.8)
plt.yscale('log')
plt.xlabel('Length of MIDI file')
plt.ylabel('Number of files')
plt.title('Distribution of MIDI file lengths')
plt.savefig(save_dir + 'notes.png')
plt.close()

# pitch histogram
plt.hist(pitches_concat, rwidth=0.8)
plt.yscale('log')
plt.xlabel('Pitch')
plt.ylabel('Number of notes')
plt.title('Distribution of pitches')
plt.savefig(save_dir + 'pitches.png')
plt.close()

# velocity histogram
plt.hist(velocities_concat, rwidth=0.8)
plt.yscale('log')
plt.xlabel('Velocity')
plt.ylabel('Number of notes')
plt.title('Distribution of velocities')
plt.savefig(save_dir + 'velocities.png')
plt.close()

#plot avg velocity
plt.hist(avg_velocities, rwidth=0.8)
plt.xlabel('Average velocity')
plt.ylabel('Number of files')
plt.title('Average velocities')
plt.savefig(save_dir + 'avg_vel.png')
plt.close()



with open('./data/stats.csv', 'w') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['', 'Min', 'Max', 'Avg', 'STD Dev'])
    writer.writerow(['Velocity', np.min(velocities_concat), np.max(velocities_concat), int(np.average(velocities_concat)), np.round(np.std(velocities_concat), decimals=2)])
    writer.writerow(['Length', np.min(lengths), np.max(lengths), int(np.average(lengths)), np.round(np.std(lengths), decimals=2)])
    writer.writerow(['Number of files: ' + str(len(data))])
