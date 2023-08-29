from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from scikeras.wrappers import KerasRegressor

def build_regressor():
    model = Sequential()
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2))
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
