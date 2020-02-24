from Agent import Agent
from DQNAgent import DQNAgent
from DQNetwork import DQNetwork

TARGET_NETWORK_UPDATE_FREQUENCY = 5


class DDQNAgent(DQNAgent):

    def __init__(self,
                 input_shape,
                 actions,
                 discount_factor,
                 replay_buffer,
                 minibatch_size,
                 logger):
        name = "ddqn"

        super(DDQNAgent, self).__init__(
            input_shape,
            actions,
            discount_factor,
            replay_buffer,
            minibatch_size,
            logger,
            name=name)

        self.target_network = DQNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
        self._reset_target_network()

    def _train(self, train_all=False):
        # if train_all:
        batch = self._replay_buffer.get_experiences()
        test = self._replay_buffer.get_test_experiences()
        # else:
            # batch = self._replay_buffer.get_new_experiences()

        # store log data
        loss, accuracy, mean_q_value, eval_loss, eval_acc = self.network.train(batch, test, self.target_network)

        # self._replay_buffer.reset_new_experiences()

        self._logger.add_loss([loss, eval_loss])
        self._logger.add_accuracy([accuracy, eval_acc])
        self._logger.add_q(mean_q_value)

        if self.rollout % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()

    def save_model(self):
        self.network.save(append='%s/model_%d.h5' % (self.name, self.rollout))
        self.target_network.save(append='%s/model_target_%d.h5' % (self.name, self.rollout))

    def load_model(self, rollout=1):
        self.network.load('%s/model_%d.h5' % (self.name, rollout))
        self.target_network.load('%s/model_target_%d.h5' % (self.name, rollout))

    def _reset_target_network(self):
        print("reset target to model")
        self.target_network.model.set_weights(self.network.model.get_weights())
