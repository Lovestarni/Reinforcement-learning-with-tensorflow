from collections import deque
from gym import spaces
import gym
import numpy as np

class ConcatObs(gym.Wrapper):
    def __init__(self, env, k) -> None:
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=self.k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=((k,) + shp), dtype=env.observation_space.dtype)
    
    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.stack(self.frames)

env = gym.make('BreakoutNoFrameskip-v4')
# add wrapper
wrapper_env = ConcatObs(env, 4)
print('The new observation space is: ', wrapper_env.observation_space)

# Reset the Env
obs = wrapper_env.reset()

# Take one step
obs, reward, done, info = wrapper_env.step(2)
print('Obs after taking a step is', obs.shape)