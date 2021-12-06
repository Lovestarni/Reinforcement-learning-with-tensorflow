import gym
import time

env = gym.make('MountainCar-v0')
# Number of steps you run the agent for
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # take random action, but you can also do something more intelligent
    # action = my_intelligent_agent_fn(obs) 
    action = env.action_space.sample()

    # apply the action, get new state and reward
    obs, reward, done, info = env.step(action)

    # render the environment
    env.render()

    # Wait a bit before the next frame unless you want to see a crazy fast video
    time.sleep(0.1)

    # If the episode is done, reset the environment
    if done:
        obs = env.reset()

# Close the environment
env.close()