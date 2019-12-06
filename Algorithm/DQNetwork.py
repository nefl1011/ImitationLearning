import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


class DQNetwork:

    def __init__(self, input_shape, action_space, discount_factor, minibatch_size):
        self.input_shape = input_shape
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size

        # DeepMind paper
        # todo paper lesen!!!

        self.model = Sequential()
        self.model.add(Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape,
                              data_format="channels_first"))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(self.action_space))
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["accuracy"])
        self.model.summary()  # prints model in console

    def train(self, batch, DQN_target):
        x_train = []
        target_train = []

        for datapoint in batch:
            x_train.append(datapoint['source'].astype(np.float64))

            next_state = datapoint['dest'].astype(np.float64)
            next_state_predicition = DQN_target.predict(next_state).ravel()
            next_q_value = np.max(next_state_predicition)

            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + self.discount_factor * next_q_value

            target_train.append(t)

        x_train = np.asarray(x_train).squeeze()
        target_train = np.asarray(target_train).squeeze()

        self.model.fit(x_train, target_train, batch_size=self.minibatch_size, nb_epoch=1)

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=""):
        f = ('model%s.h5' % append) if filename is None else filename
        self.model.save_weights(f)

    def load(self, path):
        self.model.load_weights(path)
