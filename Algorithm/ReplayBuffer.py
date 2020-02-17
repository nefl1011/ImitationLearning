import numpy as np
from random import randrange

ACTION_SPACE = 18


class ReplayBuffer:

    def __init__(self,
                 replay_memory_size,
                 minibatch_size):

        self._replay_memory_size = replay_memory_size
        self._minibatch_size = minibatch_size

        self._experiences = []

    def get_experiences_length(self):
        return len(self._experiences)

    def get_experiences(self):
        return self._experiences

    # def get_new_experiences(self):
    # return self._new_experiences

    # def reset_new_experiences(self):
    # self._new_experiences = []

    def add_experience(self, source, action, reward, final, initial=False):
        if len(self._experiences) >= self._replay_memory_size:
            self._experiences.pop(0)

        if initial:
            state = [source, source, source, source]
            dest = state
            action = 0
            reward = 0
            final = False
        else:
            indx = len(self._experiences) - 1
            d = self._experiences[indx]
            state = d['dest'][0]
            dest = [state[1], state[2], state[3], source]
        """
        if len(self._experiences) >= 4 and reward == 0:
            last_rewards = 0
            for exp in self._experiences[len(self._experiences) - 3:]:
                last_rewards += exp['reward']
            if last_rewards > 0:
                reward = 1
        """
        experience = {'source': np.asarray([state]),
                      'action': action,
                      'reward': reward,
                      'dest': np.asarray([dest]),
                      'final': final}

        self._experiences.append(experience)

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
        indx = len(self._experiences) - 1
        d1 = self._experiences[indx]
        return d1['source']

    def get_last_skipped_next(self):
        indx = len(self._experiences) - 1
        d1 = self._experiences[indx]
        return d1['dest']

    def get_last_experience(self):
        indx = len(self._experiences) - 1
        return self._experiences[indx]

    def save_experiences(self):
        np.save("data/exp/experiences.npy", self._experiences)

    def reset_experiences(self):
        self._experiences = []

    def load_experiences(self):
        try:
            self._experiences = np.load("data/exp/experiences.npy", allow_pickle=True).tolist()
        except IOError as io_err:
            print("Can't load experience file. Maybe not created yet.")
