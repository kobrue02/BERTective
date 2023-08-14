from keras import Sequential
from keras.layers import Dense, Activation, Bidirectional, LSTM, CuDNNLSTM, Embedding

## hyper parameters
batch_size = 32
embedding_dims = 1560 # Length of the token vectors
filters = 250 # number of filters in your Convnet
kernel_size = 6 # a window size of 3 tokens
hidden_dims = 250 #number of neurons at the normal feedforward NN
epochs = 2

model = Sequential()
model.add(Embedding(
  input_dim=44, 
  output_dim=3, 
  input_length=max_len))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

