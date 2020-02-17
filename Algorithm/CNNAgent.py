import numpy as np

from Agent import Agent
from CNN import CNN


class CNNAgent(Agent):

    def __init__(self,
                 input_shape,
                 actions,
                 replay_buffer,
                 minibatch_size,
                 logger,
                 name="cnn"):
        self.input_shape = input_shape
        self.action_space = actions
        self.minibatch_size = minibatch_size

        self.network = CNN(self.input_shape, self.action_space, self.minibatch_size)

        super(CNNAgent, self).__init__(
            logger,
            replay_buffer,
            name=name)

    def _train(self, train_all=False):
        # if train_all:
        batch = self._replay_buffer.get_experiences()
        # else:
            # batch = self._replay_buffer.get_new_experiences()

        # store log data
        loss, accuracy, eval_loss, eval_acc = self.network.train(batch)

        # self._replay_buffer.reset_new_experiences()

        self._logger.add_loss([loss, eval_loss])
        self._logger.add_accuracy([accuracy, eval_acc])

    def get_action(self, state):
        return np.argmax(self.network.predict(state))

    def _get_action_confidence(self, state):
        values = self.network.predict(state)[0]
        idxs = np.argwhere(values == np.max(values)).ravel()
        return np.random.choice(idxs)

    def save_model(self):
        self.network.save(append='%s/model_%d.h5' % (self.name, self.rollout))

    def load_model(self, rollout=1):
        self.network.load('%s/model_%d.h5' % (self.name, rollout))
