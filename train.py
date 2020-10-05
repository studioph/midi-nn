import utils
from generator import KerasBatchGenerator
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from model import create_model
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# utils.checkGPU()
data_file = 'data/train_test.npy'
train_mapping, test_mapping, x_train, x_test, y_train, y_test = np.load(data_file, allow_pickle=True)

num_epochs = 10
hidden_size = 100
batch_size = 10
seq_length = hidden_size

y_train = np.array([utils.encode_velocities(tensor) for tensor in y_train])
y_test = np.array([utils.encode_velocities(tensor) for tensor in y_test])

# train_generator = KerasBatchGenerator(x_train, y_train, batch_size, seq_length)
# test_generator = KerasBatchGenerator(x_test, y_test, batch_size, seq_length)

model = create_model(hidden_size)
checkpointer = ModelCheckpoint(filepath='model-{epoch:02d}.hdf5', verbose=1)
model.fit(x=x_train, y=y_train, 
            batch_size=batch_size, epochs=num_epochs,
            validation_data=(x_test, y_test), validation_batch_size=1,
            callbacks=[checkpointer])