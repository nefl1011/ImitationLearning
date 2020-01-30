import math
import os
from abc import ABC
from random import randrange

import numpy as np
from sklearn import svm

from Agent import Agent
from CNN import CNN
from DQNetwork import DQNetwork

SAVE_INTERVAL = 10

class DQNAgent(Agent):

    def __init__(self,
                 input_shape,
                 actions,
                 discount_factor,
                 replay_buffer,
                 minibatch_size,
                 logger):
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size

        self.rollout = 0
        self._name = "DQNAgent"

        self.network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
        self.load_model()
        super(DQNAgent, self).__init__(
            logger,
            replay_buffer,
            name=self._name)

    def get_action(self, state):
        return np.argmax(self.network.predict(state))

    def _get_action_confidence(self, state):
        q_values = self.network.predict(state)[0]
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        max_q = np.random.choice(idxs)
        # boltzmann
        return np.exp(q_values[max_q]) / np.sum(np.exp(q_values))

    def _train(self, train_all=False):
        self.rollout += 1
        if train_all:
            batch = self._replay_buffer.get_experiences()
        else:
            batch = self._replay_buffer.get_new_experiences()

        # store log data
        loss, accuracy, mean_q_value, eval_loss, eval_acc = self.network.train(batch)

        self._replay_buffer.reset_new_experiences()

        self._logger.add_loss(loss)
        self._logger.add_accuracy(accuracy)
        self._logger.add_q(mean_q_value)
        self._logger.add_eval_loss(eval_loss)
        self._logger.add_eval_acc(eval_acc)

        if self.rollout % SAVE_INTERVAL == 0:
            self.save_model()

    def save_model(self):
        self.network.save(append=self._name)

    def load_model(self):
        loadstring = 'data/models/model_%s.h5' % self._name
        self.network.load(loadstring)
