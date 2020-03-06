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

CLIPPING_VAL = 0.2
CRITIC_DISCOUNT = 0.5
ENTROPY_BETA = 0.001

def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - CLIPPING_VAL, max_value=1 + CLIPPING_VAL) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = CRITIC_DISCOUNT * critic_loss + actor_loss - ENTROPY_BETA * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss


class ActorNetwork(Network):

    def __init__(self, input_shape, action_space, discount_factor, minibatch_size, mode='ppo'):
        super(ActorNetwork, self).__init__(
            input_shape,
            action_space)
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.mode = mode

        self.dummy_n = np.zeros((1, 1, self.action_space))
        self.dummy_1 = np.zeros((1, 1, 1))

        state_input = Input(shape=input_shape, name='state_input')
        oldpolicy_probs = Input(shape=(1, self.action_space,), name='oldpolicy_probs')
        advantages = Input(shape=(1, 1,), name='advantages')
        rewards = Input(shape=(1, 1,), name='rewards')
        values = Input(shape=(1, 1,), name='values')

        x = Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=input_shape,
                   data_format="channels_first")(state_input)
        x = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=input_shape,
                   data_format="channels_first")(x)
        x = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=input_shape,
                   data_format="channels_first")(x)
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        out_actions = Dense(self.action_space, activation='softmax')(x)

        self.model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values], outputs=[out_actions])
        self.model.compile(optimizer=Adam(lr=1e-4),
                           loss=[ppo_loss(
                               oldpolicy_probs=oldpolicy_probs,
                               advantages=advantages,
                               rewards=rewards,
                               values=values)], metrics=["accuracy"])

    def train(self, batch, target_network=None):
        pass

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict([state, self.dummy_n, self.dummy_1, self.dummy_1, self.dummy_1])

    def load(self, path):
        if os.path.exists('data/%s' % path):
            self.model.load_weights('data/%s' % path, by_name=True)
