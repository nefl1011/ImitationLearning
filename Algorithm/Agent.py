import math
import os
from random import randrange

import numpy as np
from sklearn import svm

from CNN import CNN
from DQNetwork import DQNetwork


class Agent:

    def __init__(self, input_shape, actions, discount_factor, minibatch_size, replay_memory_size, logger, network="CNN"):
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.replay_memory_size = replay_memory_size

        self.logger = logger

        self.mode = network

        self.experiences = []
        self.new_experiences = []
        self.epochs = 0

        if self.mode is "DQN":
            self.network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
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
            batch = self.experiences
        else:
            batch = self.new_experiences

        # store log data
        loss, accuracy, mean_q_value, eval_loss, eval_acc = self.network.train(batch)

        self.new_experiences = []

        self.logger.add_loss(loss)
        self.logger.add_accuracy(accuracy)
        self.logger.add_q(mean_q_value)
        self.logger.add_eval_loss(eval_loss)
        self.logger.add_eval_acc(eval_acc)

    def evaluate_reward(self, r):
        self.logger.add_reward(r)


    def evaluate_score(self, s):
        self.logger.add_score(s)

    def get_tau_confidence(self):
        if len(self.experiences) == 0:
            return math.inf
        batch = self.sample_batch()

        # create batch of wrong_classified_confidence
        wrong_classified = []
        for datapoint in batch:
            action = self.get_action(datapoint['source'].astype(np.float64))
            if action != datapoint['action']:
                wrong_classified.append(self.get_action_confidence(datapoint['source'].astype(np.float64)))
        if len(wrong_classified) == 0:
            return 0

        self.logger.add_t_conf(np.mean(wrong_classified))
        return np.mean(wrong_classified)

    def save_model(self):
        safestring = "_%d_%s" % (self.epochs, self.mode)
        self.network.save(append="")

    def load_model(self, path):
        self.network.load(path)

    def save_experiences(self, iteration, pretrains=False):
        if pretrains:
            safestring = "_pretraining_%d_%s" % (iteration, self.mode)
        else:
            safestring = "_%d_%s" % (iteration, self.mode)
        np.save("data/experiences/experiences%s.npy" % safestring, self.experiences)

    def load_experiences(self, load_string):
        self.experiences = np.load(load_string, allow_pickle=True).tolist()
