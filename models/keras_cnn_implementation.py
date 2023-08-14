from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Embedding
from keras.preprocessing import sequence

## hyper parameters
batch_size = 32
embedding_dims = 1560 # Length of the token vectors
filters = 250 # number of filters in your Convnet
hidden_dims = 250 #number of neurons at the normal feedforward NN
epochs = 2

model = Sequential()
model.add(Conv1D())
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

