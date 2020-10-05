import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Activation, TimeDistributed, Dropout

def create_model(hidden_size, input_size, use_dropout=False):
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(128))) # 128 possible velocities
    model.add(Activation('softmax'))