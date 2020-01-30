import os
from random import randrange

import tensorflow as tf
import datetime

import numpy as np
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation
from keras.callbacks import TensorBoard

class DQNetwork:

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
        self.model.add(Dense(self.action_space))
        self.model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=["accuracy"])
        # self.model.summary()

    def train(self, batch):
        x_train = []
        target_train = []
        q_values = []

        x_test = []
        target_test = []

        for datapoint in batch:
            rand_number = randrange(0, 101)
            if rand_number > 10:
                x_train.append(datapoint['source'].astype(np.float64))
            else:
                x_test.append(datapoint['source'].astype(np.float64))

            next_state = datapoint['dest'].astype(np.float64)
            next_state_predicition = self.model.predict(next_state).ravel()
            next_q_value = np.max(next_state_predicition)
            q_values.append(next_q_value)

            t = list(self.predict(datapoint['source'])[0])
            if datapoint['final']:
                t[datapoint['action']] = datapoint['reward']
            else:
                t[datapoint['action']] = datapoint['reward'] + self.discount_factor * next_q_value

            if rand_number > 10:
                target_train.append(t)
            else:
                target_test.append(t)

        x_train = np.asarray(x_train).squeeze()
        target_train = np.asarray(target_train).squeeze()
        x_test = np.asarray(x_test).squeeze()
        target_test = np.asarray(target_test).squeeze()

        # log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        fit = self.model.fit(x_train, target_train, batch_size=self.minibatch_size, epochs=1)

        eval_results = self.model.evaluate(x_test, target_test)

        loss = fit.history["loss"][0]
        accuracy = fit.history["accuracy"][0]
        return loss, accuracy, np.mean(next_q_value), eval_results[0], eval_results[1]

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=""):
        f = ('data/models/model_%s.h5' % append) if filename is None else filename
        self.model.save(f)

    def load(self, path):
        if os.path.exists(path):
            self.model = load_model(path)
