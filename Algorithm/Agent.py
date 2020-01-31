import math
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
        self.name = name
        self._t_conf = math.inf
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
    def load_model(self):
        pass

    def train(self, train_all=False):
        self._train(train_all=train_all)
        self._t_conf = self.get_tau_confidence()
        self._logger.add_t_conf(self._t_conf)

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
        if len(wrong_classified) == 0:
            return 0

        return np.mean(wrong_classified)

    def agent_is_confident(self, state):
        action = self.get_action(state)
        conf = self._get_action_confidence(state)
        t_conf = self._t_conf
        print("Get action: %d with confidence: %f. t_conf is %f" % (action, conf, t_conf))
        return t_conf < conf
