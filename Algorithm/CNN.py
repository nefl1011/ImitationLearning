import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation


class CNN:

    def __init__(self, input_shape, action_space, discount_factor, minibatch_size):
        self.input_shape = input_shape
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size

        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(self.action_space, activation="softmax")) # classifier problem
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        # self.model.summary()  # prints model in console

    def train(self, batch):
        x_train = []
        y_train = []

        for datapoint in batch:
            x_train.append(datapoint['source'].astype(np.float64))

            t = list([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            t[datapoint['action']] = 1.
            y_train.append(t)

        x_train = np.asarray(x_train).squeeze()
        y_train = np.asarray(y_train).squeeze()

        self.model.fit(x_train, y_train, batch_size=self.minibatch_size, epochs=1)

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=""):
        f = ('model%s.h5' % append) if filename is None else filename
        self.model.save_weights(f)

    def load(self, path):
        self.model.load_weights(path)
