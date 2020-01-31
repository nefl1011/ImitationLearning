import os
from random import randrange

import numpy as np
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation

from Network import Network


class CNN(Network):

    def __init__(self, input_shape, action_space, minibatch_size):
        super(CNN, self).__init__(
            input_shape,
            action_space)
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

    def train(self, batch, target_network=None):
        x_train = []
        target_train = []

        x_test = []
        target_test = []

        for datapoint in batch:
            rand_number = randrange(0, 101)
            if rand_number > 10:
                x_train.append(datapoint['source'].astype(np.float64))
                target_train.append(datapoint['action'].astype(np.float64))
            else:
                x_test.append(datapoint['source'].astype(np.float64))
                target_test.append(datapoint['action'].astype(np.float64))

        x_train = np.asarray(x_train).squeeze()
        target_train = np.asarray(target_train).squeeze()
        x_test = np.asarray(x_test).squeeze()
        target_test = np.asarray(target_test).squeeze()

        fit = self.model.fit(x_train, target_train, batch_size=self.minibatch_size, epochs=1)

        eval_results = self.model.evaluate(x_test, target_test)

        loss = fit.history["loss"][0]
        accuracy = fit.history["accuracy"][0]
        return loss, accuracy, eval_results[0], eval_results[1]
