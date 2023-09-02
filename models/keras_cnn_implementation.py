from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM, Conv1D, MaxPooling1D, Input, TimeDistributed
from keras.optimizers import SGD, Adadelta, RMSprop, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.losses import SparseCategoricalCrossentropy

def multi_class_prediction_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Flatten(input_shape=n_inputs))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.25))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(
        loss=SparseCategoricalCrossentropy(
        from_logits=True), 
        optimizer=RMSprop(
        lr=0.001, 
        rho=0.9, 
        epsilon=None, 
        decay=0.0),
        metrics=['accuracy'])
    return model

def binary_prediction_model(n_inputs):
    model = Sequential()
    model.add(Flatten(input_shape=(n_inputs,)))
    model.add(Dense(n_inputs, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    model.compile(
        loss='binary_crossentropy', 
        optimizer=SGD(learning_rate=0.01),
        metrics=['accuracy'])
    return model

def rnn_model(n_inputs: tuple, n_outputs):
    model = Sequential()
    model.add(Input(shape=n_inputs))
    model.add(TimeDistributed(Conv1D(32, (4), activation='relu', data_format='channels_last')))
    model.add(TimeDistributed(MaxPooling1D((2))))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(
        loss=SparseCategoricalCrossentropy(
        from_logits=True), 
        optimizer=RMSprop(
        lr=0.001, 
        rho=0.9, 
        epsilon=None, 
        decay=0.0),
        metrics=['accuracy'])
    return model