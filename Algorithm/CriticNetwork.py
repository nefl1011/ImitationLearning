import os
from random import randrange

import tensorflow as tf
import datetime

import numpy as np
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation
from keras.callbacks import TensorBoard
from keras import backend as K, Input, Model
from keras.optimizers import Adam

from Network import Network


class CriticNetwork(Network):

    def __init__(self, input_shape, action_space, discount_factor, minibatch_size, mode='dqn'):
        super(CriticNetwork, self).__init__(
            input_shape,
            action_space)
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.mode = mode

        state_input = Input(shape=input_shape)
        x = Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")(state_input)
        x = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")(x)
        x = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape, data_format="channels_first")(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        out_actions = Dense(1)(x)

        self.model = Model(inputs=[state_input], outputs=[out_actions])
        self.model.compile(optimizer=Adam(lr=1e-4), loss="mse", metrics=["accuracy"])

    def train(self, batch, target_network=None):
        pass
