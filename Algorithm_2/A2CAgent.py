import os

from keras import Input
from keras import backend as K
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
import numpy as np

from Agent import Agent


class A2CAgent(Agent):

    def __init__(self,
                 input_shape,
                 actions,
                 discount_factor,
                 replay_buffer,
                 minibatch_size,
                 logger,
                 alpha=0.00001,
                 beta=0.00005,
                 gamma=0.99,
                 name="a2c"):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.actor_loss = 0.0
        self.critic_loss = 0.0

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        super(A2CAgent, self).__init__(
            logger,
            replay_buffer,
            name=name)

    def build_actor_critic_network(self):
        state_input = Input(shape=self.input_shape, name='state_input')
        delta = Input(shape=[1], name='delta')

        c1 = Conv2D(32, 8, strides=(4, 4), padding="valid", activation="relu", input_shape=self.input_shape,
                    data_format="channels_first")(state_input)
        c2 = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", input_shape=self.input_shape,
                    data_format="channels_first")(c1)
        c3 = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=self.input_shape,
                    data_format="channels_first")(c2)
        f = Flatten()(c3)
        d = Dense(512, activation="relu")(f)
        probs = Dense(18, activation='softmax')(d)  # action_space
        values = Dense(1, activation='linear')(d)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1 - 1e-8)
            log_lik = y_true * K.log(out)

            return K.sum(-log_lik * delta)

        actor = Model(input=[state_input, delta], output=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(input=[state_input], output=[values])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        policy = Model(input=[state_input], output=[probs])
        return actor, critic, policy

    def get_action(self, state):
        state = state.astype(np.float64)
        probabilites = self.policy.predict(state)[0]
        action = np.argmax(probabilites)
        return action

    def get_random_action(self, state):
        state = state.astype(np.float64)
        probabilites = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilites)
        return action

    def _get_action_confidence(self, state):
        state = state.astype(np.float64)
        probabilites = self.policy.predict(state)[0]
        return np.max(probabilites)

    def train(self, train_all=False):
        self._train(train_all=train_all)

    def _train(self, train_all=False):
        batch = self._replay_buffer.get_last_experience()

        state = batch['source'].astype(np.float64)
        next_state = batch['dest'].astype(np.float64)

        critic_value_next = self.critic.predict(next_state)
        critic_value = self.critic.predict(state)

        target = batch['reward'] + self.gamma * critic_value_next * (1 - int(batch['final']))
        delta = target - critic_value

        action = batch['action']
        action_onehot = np.zeros([1, self.action_space])
        action_onehot[np.arange(1), action] = 1.0

        actor_fit = self.actor.fit([state, delta], action_onehot, verbose=0)
        critic_fit = self.critic.fit(state, target, verbose=0)
        self.actor_loss = actor_fit.history["loss"][0]
        self.critic_loss = critic_fit.history["loss"][0]


    def save_model(self):
        self.actor.save('data/%s/actor_%d.h5' % (self.name, self.rollout))
        self.critic.save('data/%s/critic_%d.h5' % (self.name, self.rollout))
        self.policy.save('data/%s/policy_%d.h5' % (self.name, self.rollout))

    def load_model(self, rollout=1):
        if os.path.exists('data/%s/actor_%d.h5' % (self.name, rollout)):
            self.actor.load_weights('data/%s/actor_%d.h5' % (self.name, rollout))
            self.critic.load_weights('data/%s/critic_%d.h5' % (self.name, rollout))
            self.policy.load_weights('data/%s/policy_%d.h5' % (self.name, rollout))

    def reset(self):
        self.save_model()
        self.rollout += 1
        self._t_conf = self.get_tau_confidence()
        self._logger.add_t_conf(self._t_conf)
        self.save_experiences()
        self._logger.add_loss([self.actor_loss, self.critic_loss])
