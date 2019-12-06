import argparse
import sys

import cv2
import gym
import time
import random

import numpy as np
from PIL import Image

from Agent import Agent

env = gym.make('Centipede-ram-v4' if len(sys.argv) < 2 else sys.argv[1])

human_agent_action = 0
expert_is_teaching = False
human_wants_restart = False
human_sets_pause = False


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', help='name of directory to save model', default='data')
    parser.add_argument('--max_to_keep', help='number of models to save', default=10, type=int)
    parser.add_argument('--logdir', help='log directory', default='log/train/bc')
    parser.add_argument('--iteration', default=int(1e3), type=int)
    parser.add_argument('--interval', help='save interval', default=int(1e2), type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=10, type=int)
    return parser.parse_args()


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    expert_is_teaching = True
    if key == 65293: human_wants_restart = True  # enter
    if key == 112: human_sets_pause = not human_sets_pause  # p
    if key == 97 or key == 65361:
        a = 4  # a
    elif key == 100 or key == 65363:
        a = 3  # d
    elif key == 119 or key == 65362:
        a = 2  # w
    elif key == 115 or key == 65364:
        a = 5  # s
    elif key == 32:
        a = 1  # space
    else:
        a = 0  # everything else

    human_agent_action = a


def key_release(key, mod):
    global human_agent_action, expert_is_teaching
    expert_is_teaching = False
    if key == 97 or key == 65361:
        a = 4  # a
    elif key == 100 or key == 65363:
        a = 3  # d
    elif key == 119 or key == 65362:
        a = 2  # w
    elif key == 115 or key == 65364:
        a = 5  # s
    elif key == 32:
        a = 1  # space
    else:
        a = 0  # everything else
    if human_agent_action == a:
        human_agent_action = 0


def rollout(env, agent):
    # initilization
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    human_wants_restart = False
    skip = 0
    total_timesteps = 0

    human_wants_restart = False
    print("rollout")
    obs = env.reset()
    agent, human = 0, 0

    while True:
        done = False
        #  t_dist = nn_tau_distance(states, obs)
        if c_p > t_conf:  # & nearest_neighbor(states, obs)[0] < nn_tau_distance(states, obs):
            agent += 1
            obs, r, done, info = env.step(a)

        else:
            # todo pause
            human += 1
            print("Expert needed")

            if not skip:
                a = human_agent_action
                total_timesteps += 1
                skip = SKIP_CONTROL
            else:
                skip -= 1

            if a != 0:
                # todo train model after each rollout
                obs, r, done, info = env.step(a)

        window_still_open = env.render()
        if window_still_open == False:
            return False
        if done:
            env.reset()
        if human_wants_restart:
            break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)

        # time.sleep(0.1) render a frame after 0.1 seconds
        time.sleep(0.025)

    print("timesteps %i" % (total_timesteps))

def preprocess_observation(obs, img_size):
    image = Image.fromarray(obs, 'RGB').convert('L').resize(img_size)
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])

def main(args):

    env = gym.make('Centipede-v4')
    env.render()
    # set key listener
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release


    print("Press keys w a s d or arrow-keys to move")
    print("Press space to shoot")
    print("No keys pressed is taking action 0 --> no action")
    print("\nGood Luck!")

    input_shape = (4, 110, 84) # formated image
    discount_factor = 0.9
    minibatch_size = 32
    replay_memory_size = 1024
    img_size = (84, 110)

    agent = Agent(input_shape, env.action_space.n, discount_factor, minibatch_size, replay_memory_size)

    max_epochs = 5
    max_episode_in_epoch = 100
    epoch = 0

    while epoch < max_epochs:
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.array([obs, obs, obs, obs])

        episode = 0

        while episode < max_episode_in_epoch:
            env.render()
            action = agent.get_action(np.asarray([current_state]))
            obs, r, done, info = env.step(action)
            obs = preprocess_observation(obs, img_size)

            next_state = np.array([current_state[1], current_state[2], current_state[3], obs])

            clipped_reward = np.clip(r, -1, 1)
            agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            current_state = next_state
            score += r
            episode += 1

        epoch += 1



if __name__ == '__main__':
    args = argparser()
    main(args)
