import numpy as np
from random import randrange

from PIL import Image
from scipy.stats import stats

ACTION_SPACE = 18


class ReplayBuffer:

    def __init__(self,
                 replay_memory_size,
                 minibatch_size):

        self._replay_memory_size = replay_memory_size
        self._minibatch_size = minibatch_size

        self._experiences = []
        self._test_experiences = []
        self._last_states = []
        self._last_actions = []
        self._last_rewards = []

    def get_experiences_length(self):
        return len(self._experiences)

    def get_experiences(self):
        return self._experiences

    def get_test_experiences(self):
        return self._test_experiences

    # def get_new_experiences(self):
    # return self._new_experiences

    # def reset_new_experiences(self):
    # self._new_experiences = []

    def add_experience(self, source, action, reward, final, initial=False, is_expert=True):
        if is_expert:
            if len(self._experiences) >= self._replay_memory_size:
                self._experiences.pop(0)

            rand_number = randrange(0, 101)

            if initial:
                self._last_states = [source, source, source, source]
                self._last_actions = [0, 0, 0, 0]
                self._last_rewards = [0, 0, 0, 0]
                state = self._last_states
                dest = state
                action = 0
                reward = 0
                final = False
            else:
                state = self._last_states
                dest = [self._last_states[1], self._last_states[2], self._last_states[3], source]
                self._last_states = dest
                # self._last_rewards = [self._last_rewards[1], self._last_rewards[2], self._last_rewards[3], reward]
                # self._last_actions = [self._last_actions[1], self._last_actions[2], self._last_actions[3], action]
                # reward = stats.mode(self._last_rewards)[0][0]
                # action = stats.mode(self._last_actions)[0][0]

            experience = {'source': np.asarray([state]),
                          'action': action,
                          'reward': reward,
                          'dest': np.asarray([dest]),
                          'final': final}

            if initial:
                self._experiences.append(experience)
                self._test_experiences.append(experience)
            elif rand_number > 10:
                self._experiences.append(experience)
            else:
                self._test_experiences.append(experience)
        else:
            print("Add agent experiences")
            self._last_rewards = [self._last_rewards[1], self._last_rewards[2], self._last_rewards[3], reward]
            self._last_actions = [self._last_actions[1], self._last_actions[2], self._last_actions[3], action]
            self._last_states = [self._last_states[1], self._last_states[2], self._last_states[3], source]

    def get_action_over_skipped_frames(self):
        if len(self._experiences) < 4:
            return 0
        return

    def sample_batch(self):
        batch = []
        for i in range(int((len(self._experiences)) / 10)):
            batch.append(self._experiences[randrange(0, len(self._experiences))])
        return batch

    def get_last_skipped(self):
        return np.asarray([self._last_states])

    def get_last_skipped_next(self):
        indx = len(self._experiences) - 1
        d1 = self._experiences[indx]
        return d1['dest']

    def get_last_experience(self):
        indx = len(self._experiences) - 1
        return self._experiences[indx]

    def save_experiences(self):
        np.save("data/exp/experiences.npy", self._experiences)
        np.save("data/exp/test_experiences.npy", self._test_experiences)

    def reset_experiences(self):
        self._experiences = []

    def load_experiences(self):
        try:
            self._experiences = np.load("data/exp/experiences.npy", allow_pickle=True).tolist()
            self._test_experiences = np.load("data/exp/test_experiences.npy", allow_pickle=True).tolist()
        except IOError as io_err:
            print("Can't load experience file. Maybe not created yet.")
