import math
import sys
from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):

    SAVE_INTERVAL = 10

    def __init__(self,
                 logger,
                 replay_buffer,
                 name=""):
        self._logger = logger
        self._replay_buffer = replay_buffer
        self.load_experiences()
        self.name = name
        self._t_conf = self.get_tau_confidence()
        self.rollout = 1
        super().__init__()

    @abstractmethod
    def _train(self, train_all=False):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def _get_action_confidence(self, state):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self, rollout=1):
        pass

    def train(self, train_all=False):
        self._train(train_all=train_all)
        self.save_model()
        self.rollout += 1
        self._t_conf = self.get_tau_confidence()
        self._logger.add_t_conf(self._t_conf)
        self.save_experiences()

    def get_tau_confidence(self):
        if self._replay_buffer.get_experiences_length() == 0:
            return math.inf
        batch = self._replay_buffer.sample_batch()

        # create batch of wrong_classified_confidence
        wrong_classified = []
        for datapoint in batch:
            action = self.get_action(datapoint['source'].astype(np.float64))
            if action != datapoint['action']:
                wrong_classified.append(self._get_action_confidence(datapoint['source'].astype(np.float64)))
            else:
                print(datapoint['action'])
        if len(wrong_classified) == 0:
            sys.exit("No more states... finished!")
            return 0

        t = np.mean(wrong_classified) - (2 * np.std(wrong_classified))
        if math.isnan(t):
            return 0.682  # std
        print("percentage of wrong classified states: %f" % (len(wrong_classified) / (self._replay_buffer.get_experiences_length() / 10)))
        print("mean: %f, std: %f" % (np.mean(wrong_classified), np.std(wrong_classified)))
        return t

    def agent_is_confident(self, state):
        action = self.get_action(state)
        conf = self._get_action_confidence(state)
        t_conf = self._t_conf
        print("Get action: %d with confidence: %f. t_conf is %f" % (action, conf, t_conf))
        return t_conf <= conf

    def get_t_conf(self):
        return self._t_conf

    def set_rollout(self, roll):
        self.rollout = roll

    def save_experiences(self):
        self._replay_buffer.save_experiences()

    def load_experiences(self):
        self._replay_buffer.load_experiences()
