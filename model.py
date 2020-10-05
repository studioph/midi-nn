from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Activation, TimeDistributed, Dropout

def create_model(hidden_size, use_dropout=True):
    model = Sequential()
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(128))) # 128 possible velocities
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())
    return model