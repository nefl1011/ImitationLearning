import argparse
import os
import sys
from random import randrange

import cv2
import gym
import time
import numpy as np
from PIL import Image

from A2CAgent import A2CAgent
from CNNAgent import CNNAgent
from DQNAgent import DQNAgent
from Logger import Logger
from DDQNAgent import DDQNAgent
from PPOAgent import PPOAgent
from ReplayBuffer import ReplayBuffer

human_agent_action = 0
expert_is_teaching = False
human_wants_restart = False
human_sets_pause = False
img_size = (84, 84)
scores = []
score = 0
frame = 0
skip_frame_rate = 4
currentsteps = 1
pause_seconds = 5
mode = 'ppo'


def argparser():
    parser = argparse.ArgumentParser(description='Algorithm to teach a CNN playing a atari game.')
    parser.add_argument('--atari_game', help='name of an atari game supported by gym', default='Centipede-v4')
    parser.add_argument('--savedir', help='name of directory to save model', default='data/models/')
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--replay_memory_size', default=7500, type=int)  # +- 10 full games
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--mode', default='a2c', type=str)
    parser.add_argument('--max_episodes', default=102, type=int)  # 101
    parser.add_argument('--max_expert_rollouts', default=1, type=int)
    parser.add_argument('--skip_frame_rate', default=4, type=int)
    parser.add_argument('--pause_gap', default=5, type=int)
    return parser.parse_args()


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    expert_is_teaching = True
    if key == 65293:
        sys.exit("Exit")  # enter

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
    global human_agent_action, expert_is_teaching, pause_seconds
    expert_is_teaching = False

    if key == 112:
        pause_seconds += 60  # p
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


def step(env, action, agent):
    global skip_frame_rate, img_size
    obs_buffer = []
    total_reward = 0.0
    done = False
    for i in range(skip_frame_rate):
        window_still_open = env.render()
        if not window_still_open:
            sys.exit("Exit")

        obs, reward, temp_done, info = env.step(action)
        obs_buffer.append(preprocess_observation(obs, img_size))
        total_reward += reward
        done = done | temp_done
        if agent == None:
            time.sleep(0.0125)

    return np.maximum(obs_buffer[skip_frame_rate - 2], obs_buffer[skip_frame_rate - 1]), total_reward, done, info


def human_expert_act(replay_buffer, env, current_state, logger, agent):
    global frame, score, skip_frame_rate
    done = False
    while not done:
        action = human_agent_action
        obs, r, done, info = step(env, action, None)

        clipped_reward = np.clip(r, -1, 1)
        replay_buffer.add_experience(obs, action, clipped_reward, done)
        agent.train(train_all=True)

        score += r
        frame += 1
        logger.add_expert_action(action)

    logger.save_expert_action()


def agent_act(agent, env, current_state, replay_buffer):
    global score, scores, frame, skip_frame_rate, mode
    done = False
    current_state = replay_buffer.get_last_skipped()

    # agent actions
    while not done and agent.agent_is_confident(current_state):
        action = agent.get_random_action(current_state)

        obs, r, done, info = step(env, action, agent)

        clipped_reward = np.clip(r, -1, 1)
        replay_buffer.add_experience(obs, action, clipped_reward, done)
        agent.train(train_all=True)

        current_state = replay_buffer.get_last_skipped()

        score += r
        frame += 1

    # reset for human expert
    if done:
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.maximum(obs, obs)
        replay_buffer.add_experience(current_state, 0, 0, False, initial=True)


def evaluate_scores(logger):
    global score, scores
    print("Total score: %d" % (score))
    # scores.append(score)
    logger.add_score(score)
    score = 0


def main(args):
    global human_agent_action, img_size, frame, score, scores, skip_frame_rate, pause_seconds, mode

    # set environment
    env = gym.make(args.atari_game)
    env.render()

    # set key listener
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("Press keys w a s d or arrow-keys to move")
    print("Press space to shoot")
    print("No keys pressed is taking action 0 --> no action")
    print("\nGood Luck!")

    input_shape = (args.skip_frame_rate, 84, 84)  # formated image
    discount_factor = args.discount_factor
    minibatch_size = args.minibatch_size
    replay_memory_size = args.replay_memory_size if mode != 'ppo' else 128
    img_size = (84, 84)
    skip_frame_rate = args.skip_frame_rate
    pause_seconds = args.pause_gap
    mode = args.mode

    logger = Logger(args.atari_game, "data/%s/log/" % mode)
    replay_buffer = ReplayBuffer(128, minibatch_size)

    agent = A2CAgent(input_shape,
                     env.action_space.n,
                     discount_factor,
                     replay_buffer,
                     minibatch_size,
                     logger)

    agent.load_model(rollout=logger.get_rollouts())
    if logger.get_rollouts() != 0:
        agent.set_rollout(logger.get_rollouts() + 1)
        start = logger.get_rollouts() + 1
    else:
        start = 0
    max_episodes = args.max_episodes
    print("previous rollouts: %d" % start)

    # start algorithm
    for episode in range(start, max_episodes):
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.maximum(obs, obs)
        replay_buffer.add_experience(current_state, 0, 0, False, initial=True)
        frame = 0
        score = 0

        # get agent action until not confident enough
        agent_act(agent, env, current_state, replay_buffer)

        # request fpr expert
        print("Need Expert Demonstration in %d seconds!" % pause_seconds)
        sec = args.pause_gap
        # if episode > 0 and episode % 20 == 0:
            # sec = 600
        while pause_seconds > 0:
            time.sleep(1)
            pause_seconds -= 1
            print(pause_seconds)
        print("Begin!")
        pause_seconds = sec

        # get expert actions until we are done
        for i in range(0, args.max_expert_rollouts):
            if i > 0:
                score = 0
                frame = 0
                obs = preprocess_observation(env.reset(), img_size)
                current_state = np.maximum(obs, obs)

            human_expert_act(replay_buffer, env, current_state, logger, agent)

            evaluate_scores(logger)

        agent.reset()


if __name__ == '__main__':
    args = argparser()
    main(args)
