import gym
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')
obs = env.reset()

# interact with the screen
# env.render(mode='human')

# observate the state by matplotlib
env_screen = env.render(mode = 'rgb_array')



# reset the environment and see the initial observation

# Observation and action space
obs_space = env.observation_space
action_space = env.action_space
print('The observation space is:', obs_space)
print('The action space is:', action_space)

# Sample a rand action from the entire action space
random_action = action_space.sample()
new_obs, reward, done, info = env.step(random_action)
print('The new observation is {}'.format(new_obs))
plt.imshow(env_screen)
# plt.pause(100)