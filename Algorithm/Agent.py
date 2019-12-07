from random import randrange

import numpy as np
from sklearn import svm

from DQNetwork import DQNetwork


class Agent:

    def __init__(self, input_shape, actions, discount_factor, minibatch_size, replay_memory_size, network="DQN"):
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_memory_size = replay_memory_size

        self.experiences = []
        self.epochs = 0

        if network is "DQN":
            self.network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
            self.target_network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
        else:
            self.network = svm.SVC(gamma='scale', probability=True)
            self.target_network = svm.SVC(gamma='scale', probability=True)

        self.target_network.model.set_weights(self.network.model.get_weights())

    def get_action(self, state):
        return np.argmax(self.network.predict(state))

    def get_action_confidence(self, state):
        return np.max(self.network.predict(state))

    def get_max_q(self, state):
        q_values = self.network.predict(state)
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        return np.random.choice(idxs)

    def add_experience(self, source, action, reward, dest, final):
        if len(self.experiences) >= self.replay_memory_size:
            self.experiences.pop(0)

        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})

    def sample_batch(self):
        batch = []
        for i in range(self.minibatch_size):
            batch.append(self.experiences[randrange(0, len(self.experiences))])
        return batch

    def train(self):
        self.epochs += 1
        batch = self.sample_batch()
        self.network.train(batch, self.target_network)

    def reset_target_network(self):
        self.target_network.model.set_weights(self.network.model.get_weights)

    def get_confidence(self):
        batch = self.sample_batch()
        wrong_classified = []
        for datapoint in batch:
            action = self.get_action(datapoint['source'].astype(np.float64))
            if action != datapoint['action']:
                wrong_classified.append(self.get_action_confidence(datapoint['source'].astype(np.float64)))
        if len(wrong_classified) == 0:
            return 0
        return np.mean(wrong_classified)

    def save(self):
        self.network.save(append='_x')
        self.target_network.save(append='_target')
