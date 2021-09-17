"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

The cart pole example. Policy is oscillated.

Using:
tensorflow 1.0
gym 0.8.0
"""

import gym
import numpy as np
# import tensorflow as tf
import torch
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

np.random.seed(2)
torch.manual_seed(2)  # reproducible

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # 非确定性算法

# Superparameters
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward
# is greater then this threshold
MAX_EP_STEPS = 1000  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

env = gym.make('CartPole-v1')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class PGNet(torch.nn.Module):
    def __init__(self, n_features, n_actions):
        super(PGNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 20)
        self.fc1_activate = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, n_actions)
        self.out_activate = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_activate(x)
        x = self.fc2(x)
        out = self.out_activate(x)
        return out

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.normal_(m.weight.data, 0, 0.1)
            torch.nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        self.n_feature = n_features
        self.n_actions = n_actions
        self.lr = lr

        self.time_step = 0
        # build net
        self.actor_net = PGNet(n_features, n_actions)
        self.loss_func = torch.nn.NLLLoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                          lr=self.lr)

    def learn(self, s, a, td):
        self.time_step += 1
        s = torch.unsqueeze(torch.FloatTensor(s))
        probs = self.actor_net.forward(s)
        loss = self.loss_func(probs, torch.LongTensor(a)) * td

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return s

    def choose_action(self, s):
        # solution_1 keep batch format
        # 这里为了保持和batch一致，用了unsqueeze增加了一个维度
        s = torch.unsqueeze(torch.FloatTensor(s))
        # 最好detach从计算图上拿下来
        prob_weights = self.actor_net.forward(s).detach()
        # 用pytorch直接生成采样或者使用numpy添加概率抽取
        m = Categorical(prob_weights)
        action = m.sample()
        # action = np.random.choice(np.arange(prob_weights.shape[1]),
        #                           p=prob_weights.ravel())
        # ravel 返回扁平表示

        # # solution_2 不保持batch的形状，更好的体现pytorch动态图的特性,但是如果这样的话
        # # softmax就不能直接把dim指定出来，要在外部指定或者使用参数传递，所以还是用第一种方法
        # # 这里为了保持和batch一致，用了unsqueeze增加了一个维度
        # s = torch.FloatTensor(s)
        # # 最好detach从计算图上拿下来
        # prob_weights = self.actor_net(self.n_feature, self.n_actions).detach()
        # # m = Categorical(prob_weights)
        # # action = m.sample()
        # action = np.random.choice(np.arange(prob_weights.shape[0]),
        #                           p=prob_weights)
        # # ravel 返回扁平表示
        return action


class QNet(torch.nn.Module):
    def __init__(self, n_features):
        super(QNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_features, 20)
        self.fc1_activate = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_activate(x)
        out = self.fc2(x)
        return out

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.normal_(m.weight.data, 0, 0.1)
            torch.nn.init.constant_(m.bias.data, 0.01)


class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.n_features = n_features
        self.lr = 0.01
        self.time_step = 0

        # build net
        self.critic_net = QNet(self.n_features)
        self.optimizer = torch.optim.Adam(self.critic_net.parameters(), self.lr)
        self.loss_func - torch.nn.MSELoss()

    def learn(self, s, r, s_):
        s, s_ = torch.unsqueeze(torch.FloatTensor(s)), torch.unsqueeze(
            torch.FloatTensor)
        v = self.critic_net.forward(s)
        v_ = self.critic_net.forward(s_)

        # loss = pass
        return td_error


actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F,
                lr=LR_C)  # we need a good teacher, so the teacher should
# learn faster than the actor


if OUTPUT_GRAPH:
    writer = SummaryWriter("logs/")
    # writer.add_graph()

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r,
                                s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a,
                    td_error)  # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  #
            # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
