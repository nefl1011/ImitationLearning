from __future__ import print_function

import sys
import gym
import time

env = gym.make('Centipede-v0' if len(sys.argv) < 2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0  # Use previous control decision SKIP_CONTROL times, that's how you can test what skip is still usable.

human_agent_action = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key == 65293: human_wants_restart = True  # enter
    if key == 112: human_sets_pause = not human_sets_pause  # p
    if key == 97:
        a = 4  # a
    elif key == 100:
        a = 3  # d
    elif key == 119:
        a = 2  # w
    elif key == 115:
        a = 5  # s
    elif key == 32:
        a = 1  # space
    else:
        key = 0  # everything else

    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a


def key_release(key, mod):
    global human_agent_action
    if key == 97:
        a = 4  # a
    elif key == 100:
        a = 3  # d
    elif key == 119:
        a = 2  # w
    elif key == 115:
        a = 5  # s
    elif key == 32:
        a = 1  # space
    else:
        key = 0  # everything else
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    while 1:
        if not skip:
            # print("taking action {}".format(human_agent_action))
            a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1

        obser, r, done, info = env.step(a)
        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open == False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()
            time.sleep(0.1)
        time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


print("ACTIONS={}".format(ACTIONS))
print("Press keys w a s d space ... ")
print("No keys pressed is taking action 0")

while 1:
    window_still_open = rollout(env)
    if window_still_open == False: break
