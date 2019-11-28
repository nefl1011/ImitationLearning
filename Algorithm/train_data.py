from __future__ import print_function

import argparse
import sys
import gym
import time

import numpy as np

from bc_model import BCModel

# Use Centipede RAM version 4 (latest)
# observation_space: 128
# action_space: 18 (6 nedded)
env = gym.make('Centipede-ram-v4' if len(sys.argv) < 2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you can test what skip is still usable.

human_agent_action = 0
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
    global human_agent_action, human_wants_restart, human_sets_pause
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

    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
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
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obs = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    states, actions = [], []
    bc_model = BCModel(env)

    while True:
        if not skip:
            # print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        # sarsa --> for markov decision process
        # sarsa = (obs, human_agent_action) # tuple

        states.append(obs)  # save observation as state
        actions.append(a)  # save action as action
        # print((obs, a))

        obs, r, done, info = env.step(a)  # update obs, r, done and info with new action

        # if r != 0:
        # print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open == False:
            # labeling --> actions onehot encoding
            bc_model.train_data(np.asarray(states, dtype=np.float32), actions, 1200, 128)

            return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)

        # time.sleep(0.1) render a frame after 0.1 seconds
        time.sleep(0.025)

    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


def main(args):
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("ACTIONS={}".format(ACTIONS))
    print("Press keys w a s d or arrow-keys to move")
    print("Press space to shoot")
    print("No keys pressed is taking action 0")
    print("\nGood Luck!")

    while 1:
        window_still_open = rollout(env)
        if window_still_open == False: break


if __name__ == '__main__':
    args = argparser()
    main(args)
