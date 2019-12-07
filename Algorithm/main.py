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

def preprocess_observation(obs, img_size):
    image = Image.fromarray(obs, 'RGB').convert('L').resize(img_size)
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])


def expert_demonstration(max_expert_rollouts, agent, img_size, update_freq, target_network_update_freq,
                         replay_start_size):
    img_size = (84, 110)
    for i in range(0, max_expert_rollouts):
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.array([obs, obs, obs, obs])
        frame = 0

        done = False
        while not done:
            env.render()
            action = human_agent_action
            obs, r, done, info = env.step(action)
            obs = preprocess_observation(obs, img_size)

            next_state = np.array([current_state[1], current_state[2], current_state[3], obs])

            clipped_reward = np.clip(r, -1, 1)
            agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            current_state = next_state
            score += r

            frame += 1
            if frame % update_freq == 0 and len(agent.experiences) >= replay_start_size:
                frame = 0
                agent.train()
                if agent.training_count % target_network_update_freq == 0 and agent.training_count >= target_network_update_freq:
                    agent.reset_target_network()

            time.sleep(0.035)

        print("Total score: %d" % (score))
        agent.train()


def main(args):
    global human_agent_action, human_wants_restart, human_sets_pause

    env = gym.make('Centipede-v4')
    env.render()
    # set key listener
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("Press keys w a s d or arrow-keys to move")
    print("Press space to shoot")
    print("No keys pressed is taking action 0 --> no action")
    print("\nGood Luck!")

    input_shape = (4, 110, 84)  # formated image
    discount_factor = 0.9
    minibatch_size = 32
    replay_memory_size = 1000000
    img_size = (84, 110)

    agent = Agent(input_shape, env.action_space.n, discount_factor, minibatch_size, replay_memory_size)

    max_episodes = 10
    max_episode_length = 10000000
    episode = 0

    # pretrain from expert
    # expert_demonstration(10, agent, img_size, 4, 10000, 50000)
    max_expert_rollouts = 5
    update_freq = 4
    replay_start_size = 1000
    target_network_update_freq = 10000

    for i in range(0, max_expert_rollouts):
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.array([obs, obs, obs, obs])
        frame = 0

        done = False
        while not done:
            env.render()
            action = human_agent_action
            obs, r, done, info = env.step(action)
            obs = preprocess_observation(obs, img_size)

            next_state = np.array([current_state[1], current_state[2], current_state[3], obs])

            clipped_reward = np.clip(r, -1, 1)
            agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

            current_state = next_state
            score += r

            frame += 1
            time.sleep(0.035)
            """
            if frame % update_freq == 0 and len(agent.experiences) >= replay_start_size:
                frame = 0
                agent.train()
                if agent.epochs % target_network_update_freq == 0 and agent.epochs % target_network_update_freq == 0:
                    agent.reset_target_network()
            else:
                time.sleep(0.035)"""

        print("Total score: %d" % (score))
        agent.train()


    while episode < max_episodes:
        env.reset()
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.array([obs, obs, obs, obs])
        frame = 0
        t_conf = agent.get_action_confidence(np.asarray([current_state]))

        while frame < max_episode_length:
            env.render()

            if t_conf >= agent.get_confidence():
                action = agent.get_action(np.asarray([current_state]))
                obs, r, done, info = env.step(action)
                obs = preprocess_observation(obs, img_size)

                next_state = np.array([current_state[1], current_state[2], current_state[3], obs])
            """
            else:

                done = False
                while not human_wants_restart:
                    print("Waiting for human interaction!")
                    if human_wants_restart:
                        human_wants_restart = False
                        break
                    else:
                        time.sleep(3)

                human_wants_restart = False
                while not done:
                    if human_wants_restart:
                        break

                    env.render()
                    action = human_agent_action
                    obs, r, done, info = env.step(action)
                    obs = preprocess_observation(obs, img_size)

                    next_state = np.array([current_state[1], current_state[2], current_state[3], obs])

                    clipped_reward = np.clip(r, -1, 1)
                    agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]),
                                         done)

                    current_state = next_state
                    score += r

                    frame += 1
                    time.sleep(0.035)

                print("Total score: %d" % (score))
                agent.train()
                agent.reset_target_network()
                t_conf = agent.get_action_confidence(np.asarray([current_state]))"""

            current_state = next_state
            score += r
            frame += 1

        episode += 1


if __name__ == '__main__':
    args = argparser()
    main(args)
