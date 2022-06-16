import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42, return_info=True)