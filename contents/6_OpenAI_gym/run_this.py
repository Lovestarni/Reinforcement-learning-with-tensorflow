import gym
from DQN_torch import DeepQNetwork
import numpy as np
from loguru import logger

env = gym.make('CartPole-v0')
env = env.unwrapped
# 解除一些可能的限制

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


def run_dqn(epoch=100):
    total_steps = 0
    avg_rewards = []
    for i_episode in range(100):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()  # 刷新环境
            action = RL.choose_action(observation)
            # Action
            # 0 left
            # 1 right

            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_
            # 细分为了更好的进行reward修改
            # x 是车辆水平位移， cart position min -4.8, max 4.8
            # x_dot 是车辆速度， cart velocity min -Inf, max Inf
            # theta 是杆垂直的角度， Pole Angle min 0.418 rad (-24 deg) max 0.418 rad (
            # 24 deg)
            # theta_dot 是杆旋转角速度, Pole Angel Velocity min -Inf, max Inf
            # r1 是车辆距离中心分数高
            # r2 是杆子垂直的角度，角度越小分数越高
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / \
                 env.theta_threshold_radians - 0.5
            reward = r1 + r2

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > 1000:
                RL.learn()

            ep_r += reward
            # 通过记录每一步得到的reward，这样就会得到average return，
            # 比cost更好的评价训练的过程
            if done:
                print('episode：', i_episode, 'ep_r: ', round(ep_r, 2),
                            ' epsilon: ', round(RL.epsilon, 2))
                avg_rewards.append(ep_r/reward)

                # if total_steps % 10 == 0:
                #     np.save(np.array(avg_rewards))
                break
            observation = observation_
            total_steps += 1


if __name__ == '__main__':
    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      hidden_size=50,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      batch_size=32,
                      output_graph=True,
                      e_greedy_increment=0.001)
    run_dqn(100)
