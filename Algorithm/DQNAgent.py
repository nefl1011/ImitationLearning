import numpy as np

from Agent import Agent
from DQNetwork import DQNetwork


class DQNAgent(Agent):

    def __init__(self,
                 input_shape,
                 actions,
                 discount_factor,
                 replay_buffer,
                 minibatch_size,
                 logger,
                 name="dqn"):
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size

        self.network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)

        super(DQNAgent, self).__init__(
            logger,
            replay_buffer,
            name=name)

    def get_action(self, state):
        return np.argmax(self.network.predict(state))

    def _get_action_confidence(self, state):
        q_values = self.network.predict(state)[0]
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        max_q = np.random.choice(idxs)
        # boltzmann
        return np.exp(q_values[max_q]) / np.sum(np.exp(q_values))
        """
        q_values = self.network.predict(state)[0]
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        max_q = np.random.choice(idxs)
        # sigmoid
        if q_values[max_q] >= 0:
            z = np.exp(q_values[max_q])
            return 1 / (1 + z)
        else:
            z = np.exp(q_values[max_q])
            return z / (1 + z)
        """

    def _train(self, train_all=False):
        # if train_all:
        batch = self._replay_buffer.get_experiences()
        # else:
            # batch = self._replay_buffer.get_new_experiences()

        # store log data
        loss, accuracy, mean_q_value, eval_loss, eval_acc = self.network.train(batch)

        # self._replay_buffer.reset_new_experiences()

        self._logger.add_loss([loss, eval_loss])
        self._logger.add_accuracy([accuracy, eval_acc])
        self._logger.add_q(mean_q_value)

    def save_model(self):
        self.network.save(append='%s/model_%d.h5' % (self.name, self.rollout))

    def load_model(self, rollout=1):
        self.network.load('%s/model_%d.h5' % (self.name, rollout))
