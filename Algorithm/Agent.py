import math
from random import randrange

import numpy as np
from sklearn import svm

from CNN import CNN
from DQNetwork import DQNetwork


class Agent:

    def __init__(self, input_shape, actions, discount_factor, minibatch_size, replay_memory_size, network="CNN"):
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_memory_size = replay_memory_size

        self.mode = network

        self.experiences = []
        self.new_experiences = []
        self.epochs = 0

        if self.mode is "DQN":
            self.network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
            self.target_network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
            self.target_network.model.set_weights(self.network.model.get_weights())
        else:
            self.network = CNN(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)



    def get_action(self, state):
        return np.argmax(self.network.predict(state))

    def get_action_confidence(self, state):
        if self.mode == "DQN":
            q_values = self.network.predict(state)[0]
            idxs = np.argwhere(q_values == np.max(q_values)).ravel()
            max_q = np.random.choice(idxs)
            # boltzmann
            return np.exp(q_values[max_q]) / np.sum(np.exp(q_values))
        return np.max(self.network.predict(state))

    def get_max_q(self, state):
        q_values = self.network.predict(state)
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        return np.random.choice(idxs)

    def add_experience(self, source, action, reward, dest, final):
        if len(self.experiences) >= self.replay_memory_size:
            self.experiences.pop(0)

        if len(self.new_experiences) >= self.replay_memory_size:
            self.new_experiences.pop(0)

        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})

        self.new_experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})

    def sample_batch(self):
        batch = []
        for i in range(self.minibatch_size):
            batch.append(self.experiences[randrange(0, len(self.experiences))])
        return batch

    def train(self, train_all=False):
        self.epochs += 1
        if train_all:
            batch = self.new_experiences
        else:
            batch = self.sample_batch()
        if self.mode == "DQN":
            self.network.train(batch, self.target_network)
        else:
            self.network.train(batch)
        self.new_experiences = []

    def reset_target_network(self):
        self.target_network.model.set_weights(self.network.model.get_weights)

    def get_tau_confidence(self):
        if len(self.experiences) == 0:
            return math.inf
        batch = self.sample_batch()
        wrong_classified = []
        # create batch of wrong_classified_confidence
        for datapoint in batch:
            action = self.get_action(datapoint['source'].astype(np.float64))
            if action != datapoint['action']:
                wrong_classified.append(self.get_action_confidence(datapoint['source'].astype(np.float64)))
        if len(wrong_classified) == 0:
            return 0
        return np.mean(wrong_classified)

    def save(self):
        self.network.save(append='_x')
        if self.mode == "DQN":
            self.target_network.save(append='_target')
