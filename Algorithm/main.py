import argparse
import os
import sys
from random import randrange

import cv2
import gym
import time
import numpy as np
from PIL import Image

from CNNAgent import CNNAgent
from DQNAgent import DQNAgent
from Logger import Logger
from DDQNAgent import DDQNAgent, TARGET_NETWORK_UPDATE_FREQUENCY
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
    parser.add_argument('--replay_memory_size', default=1024, type=int)  # +- 10 full games
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--mode', default='ddqn', type=str)
    parser.add_argument('--max_episodes', default=513, type=int)  # 101
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

#number = 0
def preprocess_observation(obs, img_size):
    #global number
    #print(number)
    #img_rgb = Image.fromarray(obs, 'RGB')
    #img_g = Image.fromarray(obs, 'RGB').convert('L')
    #img_c = Image.fromarray(obs, 'RGB').convert('L').crop((10, 10, 150, 210))
    image = Image.fromarray(obs, 'RGB').convert('L').crop((10, 10, 150, 210)).resize(img_size)
    #print()
    #img_rgb.save("data/img/rgb/img_%d.jpg" % number)
    #img_g.save("data/img/greyscale/img_%d.jpg" % number)
    #img_c.save("data/img/cropped/img_%d.jpg" % number)
    #image.save("data/img/resize/img_%d.jpg" % number)
    #number += 1
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])


def step(env, action, agent):
    global skip_frame_rate, img_size, human_agent_action, frame
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

        time.sleep(0.025)

    diff = (obs_buffer[0] - obs_buffer[1]) + (obs_buffer[1] - obs_buffer[2]) + (obs_buffer[2] - obs_buffer[3])

    return np.maximum(obs_buffer[2], obs_buffer[3]), total_reward, done, info


def expert_pretrain(replay_buffer, logger, env, agent):
    global frame, score, skip_frame_rate, human_agent_action
    done = False
    obs = preprocess_observation(env.reset(), img_size)
    current_state = np.maximum(obs, obs)
    replay_buffer.add_experience(current_state, 0, 0, False, initial=True)

    while not done:
        action = human_agent_action
        obs, r, done, info = step(env, action, None)

        clipped_reward = np.clip(r, -1, 1)
        replay_buffer.add_experience(obs, action, clipped_reward, done)

        score += r
        frame += 1
        logger.add_expert_action(action)

    logger.save_expert_action()
    agent.train(train_all=True)


def human_expert_act(replay_buffer, env, current_state, logger, agent):
    global frame, score, skip_frame_rate, human_agent_action
    done = False
    # i = 0
    while not done and not agent.agent_is_confident(replay_buffer.get_last_skipped()):  # or i < 16:
        action = human_agent_action
        obs, r, done, info = step(env, action, None)
        clipped_reward = np.clip(r, -1, 1)

        print("expert chose action: %d" % human_agent_action)
        replay_buffer.add_experience(obs, action, clipped_reward, done)

        score += r
        frame += 1
        logger.add_expert_action(action)
        # i += 1

    # logger.save_expert_action()
    return done


def agent_act(agent, env, logger, replay_buffer):
    global score, scores, frame, skip_frame_rate, mode
    done = False
    current_state = replay_buffer.get_last_skipped()

    # agent actions
    while not done and agent.agent_is_confident(current_state):
        action = agent.get_action(current_state)
        logger.add_agent_action(action)
        obs, r, done, info = step(env, action, agent)

        clipped_reward = np.clip(r, -1, 1)

        print("expert chose action: %d" % human_agent_action)
        replay_buffer.add_experience(obs, human_agent_action, clipped_reward, done)

        current_state = replay_buffer.get_last_skipped()

        score += r
        frame += 1
        """
        if done:
            score = 0
            frame = 0
            obs = env.reset()
            obs = preprocess_observation(obs, img_size)
            obs = (obs - obs) + (obs - obs) + (obs - obs)
            replay_buffer.add_experience(obs, 0, 0, False, initial=True, is_expert=False)
            current_state = replay_buffer.get_last_skipped()
        """
    return done


def evaluate_scores(logger):
    global score, scores
    print("Total score: %d" % (score))
    # scores.append(score)
    logger.add_score(score)
    score = 0


def wrap_reward(action, reward):
    if reward == 1.0:
        return 1.0
    if action == 0 or action == 3 or action == 4:
        return 0.25
    elif action == 2 or action == 5:
        return 0.05

    return 0.0


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
    replay_memory_size = args.replay_memory_size
    img_size = (84, 84)
    skip_frame_rate = args.skip_frame_rate
    pause_seconds = args.pause_gap
    mode = "conf_dagger"

    logger = Logger(args.atari_game, "data/%s/log/" % mode)
    replay_buffer = ReplayBuffer(replay_memory_size, minibatch_size)

    print("Using DDQN agent")
    agent = DDQNAgent(input_shape,
                      6,
                      discount_factor,
                      replay_buffer,
                      minibatch_size,
                      logger,
                      mode)
    """
    expert = DDQNAgent(input_shape,
                      6,
                      discount_factor,
                      replay_buffer,
                      minibatch_size,
                      logger,
                      mode)
    expert.load_model(rollout=0)
    """
    expert = None
    if expert == None:
        env.unwrapped.viewer.window.on_key_press = key_press
        env.unwrapped.viewer.window.on_key_release = key_release

    agent.load_model(rollout=logger.get_rollouts())
    agent.set_tau_conf()
    if logger.get_rollouts() != 0:
        agent.set_rollout(logger.get_rollouts() + 1)
        start = logger.get_rollouts() + 1
    else:
        start = 1
    max_episodes = args.max_episodes
    print("previous rollouts: %d" % start)

    # start algorithm
    for episode in range(start, max_episodes):
        obs = env.reset()
        done = False
        initial = True
        frame = 0
        score = 0
        obs = preprocess_observation(obs, img_size)
        current_state = (obs - obs) + (obs - obs) + (obs - obs)

        replay_buffer.add_experience(obs, 0, 0, False, initial=initial)

        # get agent action until not confident enough
        while not done:
            if agent.agent_is_confident(replay_buffer.get_last_skipped()):
                action = agent.get_action(replay_buffer.get_last_skipped())
                print("agent action: %d" % action)
                logger.add_agent_action(action)
                is_expert_action = False
            else:
                log_action = agent.get_action(replay_buffer.get_last_skipped())
                logger.add_agent_action(log_action)
                if expert == None:
                    action = human_agent_action
                else:
                    action = expert.get_action(replay_buffer.get_last_skipped())
                    human_agent_action = action
                is_expert_action = True

            obs, r, done, info = step(env, action, None)
            clipped_reward = np.clip(r, -1, 1)

            print("expert chose action: %d" % human_agent_action)
            replay_buffer.add_experience(obs, human_agent_action, clipped_reward, done)
            logger.add_expert_action(human_agent_action)
            score += r
            frame += 1

        evaluate_scores(logger)
        logger.save_expert_action()
        logger.save_agent_action()

        # train additional experience
        window_still_open = env.render()
        if window_still_open:  # and episode % 10 == 0:
            agent.train(train_all=True)


if __name__ == '__main__':
    args = argparser()
    main(args)
