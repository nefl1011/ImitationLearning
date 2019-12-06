from __future__ import print_function

import argparse
import sys
import gym
import time
import random

import numpy as np

import bc_model
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

    if a <= 0 or a >= ACTIONS: return
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
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


def nn_tau_distance(states, current_state):
    ix = nearest_neighbor(states, current_state)
    return np.mean(ix) * 3  # paper tau_distance


def nearest_neighbor(states, current_state):
    ds = np.sum((states[:len(states)] - current_state) ** 2, axis=1)  # L2 distance
    return np.argsort(ds)  # sorts ascending by distance


def tau_confidence(states, actions, bc_model):
    training_states_set = []
    training_actions_set = []
    test_states_set = []
    test_actions_set = []

    # devide into training_set and test_set
    for i in range(0, len(states) - 1):
        x = random.random()
        s = states[i]
        a = actions[i]
        if x < 0.5:
            training_states_set.append(s)
            training_actions_set.append(a)
        else:
            test_states_set.append(s)
            test_actions_set.append(a)

    # get misclassified
    misclassified = []
    for cx in range(0, len(test_actions_set) - 1):
        action_p, c_p = bc_model.get_predicted_action_and_probability(test_states_set[cx])
        #  print("a_p: %d, test: %d, c: %f" %(action_p, test_actions_set[cx], c_p))

        if test_actions_set[cx] != action_p:
            misclassified.append(c_p)

    return np.mean(misclassified)


def initialization(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    states, actions = [], []
    bc_model = BCModel(env)

    t_conf = np.math.inf
    t_dist = 0

    obs = env.reset()

    skip = 0
    while True:
        if not skip:
            a = human_agent_action
            skip = SKIP_CONTROL
        else:
            skip -= 1

        states.append(obs)
        actions.append(a)
        obs, r, done, info = env.step(a)

        env.render()
        if human_wants_restart:
            break

        time.sleep(0.025)

    bc_model.train_data(np.asarray(states, dtype=np.float32), actions, 1200, 128)
    t_conf = tau_confidence(states, actions, bc_model)
    return states, actions, bc_model, t_conf, t_dist


def rollout(env):
    # initilization
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    human_wants_restart = False
    skip = 0
    total_timesteps = 0

    states, actions, bc_model, t_conf, t_dist = initialization(env)
    human_wants_restart = False
    print("rollout")
    obs = env.reset()
    agent, human = 0, 0

    while True:
        done = False
        a_p, c_p = bc_model.get_predicted_action_and_probability(obs)
        #  t_dist = nn_tau_distance(states, obs)
        if c_p > t_conf:  # & nearest_neighbor(states, obs)[0] < nn_tau_distance(states, obs):
            agent += 1
            a = a_p
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
                states.append(obs)  # save observation as state
                actions.append(a)  # save action as action
                # todo train model after each rollout
                bc_model.train_data(np.asarray(states, dtype=np.float32), actions, 1200, 128)
                t_conf = tau_confidence(states, actions, bc_model)
                obs, r, done, info = env.step(a)

        window_still_open = env.render()
        if window_still_open == False:
            # labeling --> actions onehot encoding
            print(human / agent)
            print(tau_confidence(states, actions, bc_model))
            # bc_model.train_data(np.asarray(states, dtype=np.float32), actions, 1200, 128)
            return False
        if done: env.reset()
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)

        # time.sleep(0.1) render a frame after 0.1 seconds
        time.sleep(0.025)

    print("timesteps %i" % (total_timesteps))


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
