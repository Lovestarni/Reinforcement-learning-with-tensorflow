import random
import gym
import time
import numpy as np

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0

class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def reward(self, reward):
        return np.clip(reward, 0, 1)

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        if action == 3:
            return random.choice([0, 1, 2])
        return action

env = gym.make('BreakoutNoFrameskip-v4')
# 层层wrapper
wrapper_env = ObservationWrapper(RewardWrapper(ActionWrapper(env)))
obs =  wrapper_env.reset()

num_episodes = 5000

for step in range(num_episodes):
    action = wrapper_env.action_space.sample()
    obs, reward, done, info = wrapper_env.step(action)

    # 判断是否有超界的取值
    if (obs > 1.0 ).any() or (obs < 0.0).any():
        print('obs out of range')
    
    if reward > 1.0 or reward < 0.0:
        print('reward out of range')
    
    wrapper_env.render()

    time.sleep(0.001)

wrapper_env.close()

print('done')