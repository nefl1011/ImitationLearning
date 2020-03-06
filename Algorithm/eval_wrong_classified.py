import csv
from random import randrange

import gym
import numpy as np

from DDQNAgent import DDQNAgent
from Logger import Logger
from ReplayBuffer import ReplayBuffer

if __name__ == '__main__':
    env = gym.make('Centipede-v4')

    input_shape = (4, 84, 84)  # formated image
    discount_factor = 0.99
    minibatch_size = 32
    replay_memory_size = 1024
    img_size = (84, 84)
    skip_frame_rate = 4
    mode = 'conf_dagger'

    logger = Logger('Centipede-v4', "data/%s/log/agent_actions_2/" % mode)
    replay_buffer = ReplayBuffer(replay_memory_size, minibatch_size)

    agent = DDQNAgent(input_shape,
                      env.action_space.n,
                      discount_factor,
                      replay_buffer,
                      minibatch_size,
                      logger,
                      mode)

    experiences = np.load("data/exp/experiences.npy", allow_pickle=True).tolist()
    target = "data/conf_dagger/log/wrong_classified.csv"

    for iter in range(301, 302):
        agent.load_model(rollout=iter)

        batch = []
        for i in range(100):
            batch.append(experiences[randrange(0, len(experiences))])

        wrong_classified = 0
        for datapoint in batch:
            state = datapoint['source'].astype(np.float64)
            agent_action = agent.get_action(state)
            if agent_action != datapoint['action']:
                wrong_classified += 1

        target_file = open(target, "a", newline='')
        with target_file:
            writer = csv.writer(target_file)
            writer.writerow([wrong_classified])
