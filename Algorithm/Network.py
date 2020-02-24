import math
import os

import numpy as np

from abc import ABC, abstractmethod
from keras.engine.saving import load_model


class Network(ABC):

    def __init__(self,
                 input_shape,
                 action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.model = None
        super().__init__()

    @abstractmethod
    def train(self, batch, test_batch, target_network=None):
        pass

    def predict(self, state):
        state = state.astype(np.float64)
        return self.model.predict(state, batch_size=1)

    def save(self, filename=None, append=""):
        f = ('data/%s' % append) if filename is None else filename
        self.model.save(f)

    def load(self, path):
        if os.path.exists('data/%s' % path):
            print(path)
            self.model = load_model('data/%s' % path)
