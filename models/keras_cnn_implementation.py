from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adadelta, RMSprop, Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.losses import SparseCategoricalCrossentropy

def build_cnn_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Flatten(input_shape=(n_inputs,)))
    #model.add(Dense(256, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True), 
        optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
        metrics=['accuracy'])
    return model
