import gym

if __name__ == "__main__":
    env = gym.make('Centipede-v0')
    print(env.action_space) #[Output: ] Discrete(18)
    print(env.observation_space) # [Output: ] Box(250, 160, 3)