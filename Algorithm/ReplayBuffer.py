from random import randrange


class ReplayBuffer:

    def __init__(self,
                 replay_memory_size,
                 minibatch_size):

        self._replay_memory_size = replay_memory_size
        self._minibatch_size = minibatch_size

        self._experiences = []
        self._new_experiences = []

    def get_experiences_length(self):
        return len(self._experiences)

    def get_experiences(self):
        return self._experiences

    def get_new_experiences(self):
        return self._new_experiences

    def reset_new_experiences(self):
        self._new_experiences = []

    def add_experience(self, source, action, reward, dest, final):
        if len(self._experiences) >= self._replay_memory_size:
            self._experiences.pop(0)

        if len(self._new_experiences) >= self._replay_memory_size:
            self._new_experiences.pop(0)

        self._experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})

        self._new_experiences.append({'source': source,
                                     'action': action,
                                     'reward': reward,
                                     'dest': dest,
                                     'final': final})

    def sample_batch(self):
        batch = []
        for i in range(self._minibatch_size):
            batch.append(self._experiences[randrange(0, len(self._experiences))])
        return batch
