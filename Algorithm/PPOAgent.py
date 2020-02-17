import numpy as np

from ActorNetwork import ActorNetwork, ppo_loss
from Agent import Agent
from CriticNetwork import CriticNetwork

GAMMA = 0.99
LAMBDA = 0.95


class PPOAgent(Agent):

    def __init__(self,
                 input_shape,
                 actions,
                 discount_factor,
                 replay_buffer,
                 minibatch_size,
                 logger,
                 name="ppo"):
        self.input_shape = input_shape
        self.action_space = actions
        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size

        self.actor = ActorNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)
        self.critic = CriticNetwork(self.input_shape, self.action_space, self.discount_factor, self.minibatch_size)

        self.states = []
        self.actions = []
        self.values = []
        self.masks = []
        self.rewards = []
        self.actions_probs = []
        self.actions_onehot = []

        super(PPOAgent, self).__init__(
            logger,
            replay_buffer,
            name=name)

    def reset_ppo_buffer(self):
        self.states = []
        self.actions = []
        self.values = []
        self.masks = []
        self.rewards = []
        self.actions_probs = []
        self.actions_onehot = []

    def add_experience(self, done, reward, action):
        state = self._replay_buffer.get_last_skipped().astype(np.float64)
        self.states.append(state[0])
        self.masks.append(not done)
        self.rewards.append(reward)
        self.actions.append(action)
        action_onehot = np.zeros(self.action_space)
        action_onehot[action] = 1
        self.actions_onehot.append(action_onehot)
        self.values.append(self.critic.predict(state))  # q_value
        self.actions_probs.append(self.actor.predict(state))

    def add_q_value(self):
        state = self._replay_buffer.get_last_skipped_next()
        self.values.append(self.critic.predict(state))  # q_value

    def get_action(self, state):
        predictions = self.actor.predict(state)  # [0]
        action = np.argmax(predictions)
        # action = np.random.choice(self.action_space, p=predictions[0, :])
        return action

    def _get_action_confidence(self, state):
        probabilites = self.actor.predict(state)[0]
        return np.max(probabilites)

    def get_advantages(slef, values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i + 1] * int(masks[i]) - values[i]
            gae = delta + GAMMA * LAMBDA * int(masks[i]) * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def _train(self, train_all=False):
        returns, advantages = self.get_advantages(self.values, self.masks, self.rewards)
        actions_onehot_reshaped = np.reshape(self.actions_onehot, newshape=(-1, 18))
        rewards_reshaped = np.reshape(self.rewards, newshape=(-1, 1, 1))
        q_vals = self.values[:-1]
        actor_fit = self.actor.model.fit(
            [self.states, self.actions_probs, advantages, rewards_reshaped, q_vals], [actions_onehot_reshaped],
            shuffle=True, verbose=True, epochs=8)

        return_vals_reshaped = np.reshape(returns, newshape=(-1, 1))
        critic_fit = self.critic.model.fit([self.states], [return_vals_reshaped], shuffle=True, verbose=True, epochs=8)

        print(actor_fit.history)
        print(critic_fit.history)

        self._logger.add_loss([actor_fit.history["loss"][0], critic_fit.history["loss"][0]])
        #self._logger.add_loss_critic([critic_fit.history["loss"][0], critic_fit.history["val_loss"][0]])
        #self._logger.add_accuracy([actor_fit.history["accuracy"][0], actor_fit.history["val_accuracy"][0]])
        #self._logger.add_accuracy_critic([critic_fit.history["accuracy"][0], critic_fit.history["val_accuracy"][0]])
        self._logger.add_q(np.mean(q_vals))
        self.reset_ppo_buffer()

    def save_model(self):
        self.actor.save(append='%s/model_actor_%d.h5' % (self.name, self.rollout))
        self.critic.save(append='%s/model_critic_%d.h5' % (self.name, self.rollout))

    def load_model(self, rollout=1):
        self.actor.load('%s/model_actor_%d.h5' % (self.name, rollout))
        self.critic.load('%s/model_critic_%d.h5' % (self.name, rollout))
