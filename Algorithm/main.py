import argparse
import os
import sys
import gym
import time
import numpy as np
from PIL import Image

from Agent import Agent

env = gym.make('Centipede-ram-v4' if len(sys.argv) < 2 else sys.argv[1])

human_agent_action = 0
expert_is_teaching = False
human_wants_restart = False
human_sets_pause = False


def argparser():
    parser = argparse.ArgumentParser(description='Algorithm to teach a CNN playing a atari game.')
    parser.add_argument('--atari_game', help='name of an atari game supported by gym', default='Centipede-v4')
    parser.add_argument('--savedir', help='name of directory to save model', default='data')
    parser.add_argument('--minibatch_size', default=32, type=int)
    parser.add_argument('--replay_memory_size', default=1000000, type=int)
    parser.add_argument('--discount_factor', default=0.99, type=float)
    parser.add_argument('--cnn_mode', default='DQN', type=str)
    parser.add_argument('--max_episodes', default=10, type=int)
    parser.add_argument('--max_pretraining_rollouts', default=1, type=int)
    parser.add_argument('--learning_yourself', default=False, type=bool)
    return parser.parse_args()


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, expert_is_teaching
    expert_is_teaching = True
    if key == 65293:
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


def main(args):
    global human_agent_action, human_wants_restart, human_sets_pause

    env = gym.make(args.atari_game)
    env.render()
    # set key listener
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    print("Press keys w a s d or arrow-keys to move")
    print("Press space to shoot")
    print("No keys pressed is taking action 0 --> no action")
    print("\nGood Luck!")

    input_shape = (4, 110, 84)  # formated image
    discount_factor = args.discount_factor
    minibatch_size = args.minibatch_size
    replay_memory_size = args.replay_memory_size
    img_size = (84, 110)
    learning_yourself = args.learning_yourself

    agent = Agent(input_shape, env.action_space.n, discount_factor, minibatch_size, replay_memory_size, network=args.cnn_mode)

    max_episodes = args.max_episodes

    # pretrain from expert
    max_expert_rollouts = args.max_pretraining_rollouts

    for i in range(0, max_expert_rollouts):
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.array([obs, obs, obs, obs])
        frame = 0

        done = False
        while not done:
            window_still_open = env.render()
            if not window_still_open:
                return

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

        if i == 5 or i == 10 or i == 100 or i % 200 == 0:
            agent.save_experiences(i)
        print("Total score: %d" % (score))


    # train pretrained session
    if max_expert_rollouts > 0:
        agent.train(train_all=True)

    # start algorithm
    episode = 0
    while episode < max_episodes:
        done = False
        env.reset()
        score = 0
        obs = preprocess_observation(env.reset(), img_size)
        current_state = np.array([obs, obs, obs, obs])
        frame = 0

        # get confidence
        t_conf = agent.get_tau_confidence()
        conf = agent.get_action_confidence(np.asarray([current_state]))
        print(t_conf)
        print("t_conf: %f and  current confidence: %f" % (t_conf, conf))

        # agent actions
        while not done and t_conf < conf:
            window_still_open = env.render()
            if not window_still_open:
                agent.save_model()
                return

            action = agent.get_action(np.asarray([current_state]))
            conf = agent.get_action_confidence(np.asarray([current_state]))
            print("Get action: %d with confidence: %f" % (action, conf))

            obs, r, done, info = env.step(action)
            obs = preprocess_observation(obs, img_size)

            next_state = np.array([current_state[1], current_state[2], current_state[3], obs])

            # lerne von dir dazu...
            if learning_yourself:
                clipped_reward = np.clip(r, -1, 1)
                agent.add_experience(np.asarray([current_state]), action, clipped_reward, np.asarray([next_state]),
                                     done)

            current_state = next_state
            score += r
            frame += 1

        # expert actions until we are done
        print("Need Expert Demonstration!")
        os.system("pause")
        time.sleep(2.5)
        print("Begin!")

        while not done:
            window_still_open = env.render()
            if not window_still_open:
                return
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

        print("Total score: %d" % (score))

        # train additional experience
        window_still_open = env.render()
        if window_still_open:
            agent.train(train_all=True)
        episode += 1


if __name__ == '__main__':
    args = argparser()
    main(args)
