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


def evaluate_agent(agent, env, logger):
    global skip_frame_rate
    for i in range(0, 30):
        env.reset()
        obs = preprocess_observation(env.reset(), img_size)
        # initial_buffer = []
        # for j in range(skip_frame_rate):
        # initial_buffer.append(obs)
        state = np.maximum(obs, obs)  # np.array(initial_buffer)
        current_state = [state, state, state, state]
        current_state = np.asarray([current_state])
        reward = 0
        # agent actions
        done = False
        while not done:
            action = agent.get_action(current_state)
            logger.add_agent_action(action)
            obs, r, done, info = step(env, action, agent)

            # next_state = np.array(next_state)
            next_state = np.asarray([[current_state[0][1], current_state[0][2], current_state[0][3], obs]])

            current_state = next_state

            reward += r
        logger.add_reward(reward)
        logger.save_agent_action()
    logger.save_agent_action_avg_std()


def main(args):
    global img_size, skip_frame_rate

    env = gym.make(args.atari_game)
    env.render()

    input_shape = (args.skip_frame_rate, 84, 84)  # formated image
    discount_factor = args.discount_factor
    minibatch_size = args.minibatch_size
    replay_memory_size = args.replay_memory_size
    img_size = (84, 84)
    skip_frame_rate = args.skip_frame_rate
    mode = 'ddqn'

    logger = Logger(args.atari_game, "data/%s/log/" % mode)
    replay_buffer = ReplayBuffer(replay_memory_size, minibatch_size)

    rollout_max = 55

    for k in range(0, 1):
        if k == 0:
            print("Using DDQN agent")
            agent = DDQNAgent(input_shape,
                              env.action_space.n,
                              discount_factor,
                              replay_buffer,
                              minibatch_size,
                              logger)

        for i in range(1, 63):
            agent.load_model(rollout=i)
            print("current iteration: %d" % i)
            evaluate_agent(agent, env, logger)


if __name__ == '__main__':
    args = argparser()
    main(args)
