import time

import gym
import numpy as np
from PIL import Image

from CNNAgent import CNNAgent
from DDQNAgent import DDQNAgent
from DQNAgent import DQNAgent
from Logger import Logger
from PPOAgent import PPOAgent
from ReplayBuffer import ReplayBuffer
from main import step, preprocess_observation, argparser

img_size = (84, 84)
skip_frame_rate = 4


def run_agent(agent, env):
    global skip_frame_rate
    for i in range(0, 100):
        env.reset()
        obs = preprocess_observation(env.reset(), img_size)
        state = np.maximum(obs, obs)
        current_state = [state, state, state, state]
        current_state = np.asarray([current_state])

        reward = 0
        done = False
        while not done:
            action = agent.get_action(current_state)
            # print(action)
            obs, r, done, info = step(env, action, agent)

            next_state = np.asarray([[current_state[0][1], current_state[0][2], current_state[0][3], obs]])

            current_state = next_state
            reward += r
            # time.sleep(0.005)



def main(args):
    global img_size, skip_frame_rate

    env = gym.make(args.atari_game)
    env.render()

    input_shape = (args.skip_frame_rate, 84, 84)
    discount_factor = args.discount_factor
    minibatch_size = args.minibatch_size
    replay_memory_size = args.replay_memory_size
    img_size = (84, 84)
    skip_frame_rate = args.skip_frame_rate
    mode = 'conf_dagger'

    logger = Logger(args.atari_game, "data/%s/log/" % mode)
    replay_buffer = ReplayBuffer(replay_memory_size, minibatch_size)

    rollout_max = 11

    agent = DDQNAgent(input_shape,
                      6,
                      discount_factor,
                      replay_buffer,
                      minibatch_size,
                      logger,
                      mode)

    agent.load_model(rollout=485)
    run_agent(agent, env)


if __name__ == '__main__':
    args = argparser()
    main(args)
