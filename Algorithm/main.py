import argparse
import os
import sys

import cv2
import gym
import time
import numpy as np
from PIL import Image

from CNNAgent import CNNAgent
from DQNAgent import DQNAgent
from Logger import Logger
from DDQNAgent import DDQNAgent
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


def argparser():
    parser = argparse.ArgumentParser(description='Algorithm to teach a CNN playing a atari game.')
    parser.add_argument('--atari_game', help='name of an atari game supported by gym', default='Centipede-v4')
    parser.add_argument('--savedir', help='name of directory to save model', default='data/models/')
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--replay_memory_size', default=5000, type=int)  # +- 10 or 20 full games
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--mode', default='ddqn', type=str)
    parser.add_argument('--max_episodes', default=31, type=int)  # 101
    parser.add_argument('--max_expert_rollouts', default=1, type=int)
    parser.add_argument('--skip_frame_rate', default=4, type=int)
    parser.add_argument('--pause_gap', default=5, type=int)
    return parser.parse_args()


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    expert_is_teaching = True
    if key == 65293:
        sys.exit("Exit")
        human_wants_restart = True  # enter

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
            time.sleep(0.025)

    return obs_buffer, total_reward, done, info


def human_expert_act(replay_buffer, env, current_state):
    global frame, score, skip_frame_rate
    done = False
    while not done:
        action = human_agent_action
        next_state, r, done, info = step(env, action, None)

        next_state = np.array(next_state)

        clipped_reward = np.clip(r, -1, 1)
        replay_buffer.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]),
                                     done)

        current_state = next_state

        score += r
        frame += 1


def agent_act(agent, env, current_state):
    global score, scores, frame, skip_frame_rate
    done = False

    # agent actions
    while not done and agent.agent_is_confident(np.asarray([current_state])):
        action = agent.get_action(np.asarray([current_state]))

        next_state, r, done, info = step(env, action, agent)

        next_state = np.array(next_state)

        current_state = next_state
        frame += 1
    # reset for human expert
    if done:
        env.reset()


def evaluate_reward(agent, env, logger):
    global score, scores, frame, skip_frame_rate
    done = False
    for _ in range(0, 10):
        env.reset()
        obs = preprocess_observation(env.reset(), img_size)
        initial_buffer = []
        for j in range(skip_frame_rate):
            initial_buffer.append(obs)
        current_state = np.array(initial_buffer)
        reward = 0
        # agent actions
        done = False
        while not done:
            action = agent.get_action(np.asarray([current_state]))
            next_state, r, done, info = step(env, action, agent)

            next_state = np.array(next_state)

            current_state = next_state

            reward += r
            frame += 1
        logger.add_reward(reward)

    # reset for human expert
    if done:
        env.reset()


def evaluate_scores(logger):
    global score, scores
    print("Total score: %d" % (score))
    # scores.append(score)
    logger.add_score(score)
    score = 0


def main(args):
    global human_agent_action, img_size, frame, score, scores, skip_frame_rate, pause_seconds

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
    replay_memory_size = args.replay_memory_size
    img_size = (84, 84)
    skip_frame_rate = args.skip_frame_rate
    pause_seconds = args.pause_gap
    mode = args.mode

    logger = Logger(args.atari_game, "data/%s/log/" % mode)
    replay_buffer = ReplayBuffer(replay_memory_size, minibatch_size)

    if mode == 'dqn':
        print("Using DQN agent")
        agent = DQNAgent(input_shape,
                         env.action_space.n,
                         discount_factor,
                         replay_buffer,
                         minibatch_size,
                         logger)
    elif mode == 'cnn':
        print("Using CNN agent")
        agent = CNNAgent(input_shape,
                         env.action_space.n,
                         replay_buffer,
                         minibatch_size,
                         logger)
    elif mode == 'ppo':
        print("Using PPO agent")
    else:
        print("Using DDQN agent")
        agent = DDQNAgent(input_shape,
                          env.action_space.n,
                          discount_factor,
                          replay_buffer,
                          minibatch_size,
                          logger)

    agent.load_model()

    max_episodes = args.max_episodes

    # start algorithm
    for episode in range(max_episodes):
        env.reset()
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        initial_buffer = []
        for j in range(skip_frame_rate):
            initial_buffer.append(obs)
        current_state = np.array(initial_buffer)
        frame = 0

        # get agent action until not confident enough
        agent_act(agent, env, current_state)

        # request fpr expert
        print("Need Expert Demonstration in %d seconds!" % pause_seconds)
        sec = args.pause_gap
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
                current_state = np.array([obs, obs, obs, obs])

            human_expert_act(replay_buffer, env, current_state)

            evaluate_scores(logger)

        # train additional experience
        window_still_open = env.render()
        if window_still_open:
            agent.train()

        # evaluate reward
        evaluate_reward(agent, env, logger)


if __name__ == '__main__':
    args = argparser()
    main(args)
