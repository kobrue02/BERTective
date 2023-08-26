from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.losses import SparseCategoricalCrossentropy

def build_cnn_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Flatten(input_shape=(27,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True), 
        optimizer='adam', 
        metrics=['accuracy'])
    return model
