import gym
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

# list of envs
num_envs = 3
envs = [lambda: gym.make('BreakoutNoFrameskip-v4') for _ in range(num_envs)]

# Vec Env
envs = SubprocVecEnv(envs)