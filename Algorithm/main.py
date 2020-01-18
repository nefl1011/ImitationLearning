import argparse
import os
import sys

import cv2
import gym
import time
import numpy as np
from PIL import Image

from Logger import Logger
from Agent import Agent

env = gym.make('Centipede-v4' if len(sys.argv) < 2 else sys.argv[1])

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


def argparser():
    parser = argparse.ArgumentParser(description='Algorithm to teach a CNN playing a atari game.')
    parser.add_argument('--atari_game', help='name of an atari game supported by gym', default='Centipede-v4')
    parser.add_argument('--savedir', help='name of directory to save model', default='data/models/model.h5')
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--replay_memory_size', default=5000, type=int)  # +- 10 or 20 full games
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--cnn_mode', default='DQN', type=str)
    parser.add_argument('--max_episodes', default=71, type=int) # 101
    parser.add_argument('--max_pretraining_rollouts', default=1, type=int)
    parser.add_argument('--skip_frame_rate', default=3, type=int)
    parser.add_argument('--pause_gap', default=5, type=int)
    parser.add_argument('--learning_yourself', default=False, type=bool)
    return parser.parse_args()


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    expert_is_teaching = True
    if key == 65293:
        sys.exit("Exit")
        human_wants_restart = True  # enter
    if key == 112:
        human_sets_pause = not human_sets_pause  # p
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


def step(env, action, agent):
    global skip_frame_rate, img_size
    obs_buffer = []
    total_reward = 0.0
    done = False
    for i in range(skip_frame_rate):
        window_still_open = env.render()
        if not window_still_open:
            if agent != None:
                agent.save_model()
            sys.exit("Exit")

        obs, reward, temp_done, info = env.step(action)
        obs_buffer.append(preprocess_observation(obs, img_size))
        total_reward += reward
        done = done | temp_done
        if agent == None:
            time.sleep(0.025)

    return obs_buffer, total_reward, done, info


def human_expert_act(agent, env, current_state):
    global frame, score, skip_frame_rate
    done = False
    while not done:
        action = human_agent_action
        next_state, r, done, info = step(env, action, None)

        next_state = np.array(next_state)

        clipped_reward = np.clip(r, -1, 1)
        agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

        current_state = next_state

        score += r
        frame += 1


def agent_act(agent, env, current_state, learning_yourself):
    global score, scores, frame, skip_frame_rate
    done = False
    # get confidence
    t_conf = agent.get_tau_confidence()
    conf = agent.get_action_confidence(np.asarray([current_state]))
    print("t_conf: %f and confidence: %f" % (t_conf, conf))

    # agent actions
    while not done and t_conf < conf:

        action = agent.get_action(np.asarray([current_state]))
        conf = agent.get_action_confidence(np.asarray([current_state]))
        print("Get action: %d with confidence: %f" % (action, conf))

        next_state, r, done, info = step(env, action, agent)

        next_state = np.array(next_state)

        clipped_reward = np.clip(r, -1, 1)
        # lerning yourself
        if learning_yourself:
            agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]), done)

        current_state = next_state
        frame += 1
    # reset for human expert
    if done:
        env.reset()

def evaluate_reward(agent, env, current_state):
    global score, scores, frame, skip_frame_rate
    reward = 0
    # agent actions
    done = False
    while not done:

        action = agent.get_action(np.asarray([current_state]))
        print("action: %d" % (action))
        next_state, r, done, info = step(env, action, agent)

        next_state = np.array(next_state)

        current_state = next_state

        reward += r
        frame += 1

    agent.evaluate_reward(reward)
    # reset for human expert
    if done:
        env.reset()


def evaluate_scores(agent):
    global score, scores
    print("Total score: %d" % (score))
    #scores.append(score)
    agent.evaluate_score(score)
    score = 0


def main(args):
    global human_agent_action, img_size, frame, score, scores, skip_frame_rate

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
    learning_yourself = args.learning_yourself
    skip_frame_rate = args.skip_frame_rate

    logger = Logger(args.atari_game, "data/log/")
    agent = Agent(input_shape, env.action_space.n, discount_factor, minibatch_size, replay_memory_size, logger,
                  network=args.cnn_mode)

    agent.load_model(args.savedir)

    #try:
        #agent.load_experiences("data/experiences/experiences_0_%s.npy" % args.cnn_mode)
        #scores = np.load("data/scores_expert.npy").tolist()
    #except IOError as io_err:
        #print("Can't load experience/score file.")


    max_episodes = args.max_episodes

    # pretrain from expert
    max_expert_rollouts = args.max_pretraining_rollouts
    for i in range(0, max_expert_rollouts):
        score = 0
        frame = 0
        obs = preprocess_observation(env.reset(), img_size)
        initial_buffer = []
        for j in range(skip_frame_rate):
            initial_buffer.append(obs)
        current_state = np.array(initial_buffer)

        human_expert_act(agent, env, current_state)

        evaluate_scores(agent)

    #agent.save_experiences(0)
    # np.save("data/scores_expert.npy", scores)

    # train pretrained session
    if max_expert_rollouts > 0:
        agent.train()

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
        agent_act(agent, env, current_state, learning_yourself)

        # request fpr expert
        print("Need Expert Demonstration in %d seconds!" % (args.pause_gap))
        sec = args.pause_gap
        while sec > 0:
            time.sleep(1)
            sec -= 1
            print(sec)
        print("Begin!")

        # get expert actions until we are done
        for i in range(0, max_expert_rollouts):
            if i > 0:
                score = 0
                frame = 0
                obs = preprocess_observation(env.reset(), img_size)
                current_state = np.array([obs, obs, obs, obs])

            human_expert_act(agent, env, current_state)

            evaluate_scores(agent)

        #agent.save_experiences(0)
        #np.save("data/scores_expert.npy", scores)

        # train additional experience
        window_still_open = env.render()
        if window_still_open:
            agent.train()

        # evaluate reward
        env.reset()
        obs = preprocess_observation(env.reset(), img_size)
        initial_buffer = []
        for j in range(skip_frame_rate):
            initial_buffer.append(obs)
        current_state = np.array(initial_buffer)
        evaluate_reward(agent, env, current_state)

        # save model
        if (episode + 2) % 10 == 0:
            agent.save_model()


if __name__ == '__main__':
    args = argparser()
    main(args)
