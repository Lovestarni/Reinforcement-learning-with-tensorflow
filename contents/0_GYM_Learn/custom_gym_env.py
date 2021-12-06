from IPython import display
from ChopperScape import *

env = ChoppeScape()
obs = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    if done == True:
        break
env.close