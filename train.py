import utils
from generator import KerasBatchGenerator
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from model import create_model

# utils.checkGPU()
data_file = 'data/train_test.npy'
train_mapping, test_mapping, x_train, x_test, y_train, y_test = np.load(data_file, allow_pickle=True)

train_generator = KerasBatchGenerator(x_train, y_train)
test_generator = KerasBatchGenerator(x_test, y_test)

num_epochs = 10

model = create_model(len(x_train))
checkpointer = ModelCheckpoint(filepath='model-{epoch:02d}.hdf5', verbose=1)
model.fit_generator(train_generator.generate(), len(x_train), num_epochs,
                        validation_data=test_generator.generate(),
                        validation_steps=len(x_test), callbacks=[checkpointer], 
                        use_multiprocessing=True)