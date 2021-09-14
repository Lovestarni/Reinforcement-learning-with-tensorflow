"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        # 是从空表开始，所以需要每次检查是否存在这个状态，不存在就加入
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            state_actions = self.q_table.loc[observation, :]
            #  如果多个值相同，随机取一个，避免每次取第一个
            action = np.random.choice(
                state_actions[state_actions == np.max(state_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        # 检查状态是否在表里存在，若不存在就创建
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0]*len(self.actions), index=self.q_table.columns, name=state))
