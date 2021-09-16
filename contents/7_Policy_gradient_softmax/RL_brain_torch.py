"""
Using:
pytorch: 1.0
gym: 0.8.0
"""

import numpy as np
# import tensorflow as tf
import torch
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

# reproducible
torch.manual_seed(1)


class Net(torch.nn.Module):
    # 离散空间采用policy gradient优化策略
    def __init__(self, configs):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(configs['input_size'], configs[
            'hidden_size'])
        self.fc1_activate = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(configs['hidden_size'], configs[
            'output_size'])
        self.out_activate = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_activate(x)
        x = self.fc2(x)
        out = self.out_activate(x)
        return out


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs, self.ep_as_prob = [], [], [], []

        self.pollicy_net = Net({
            'input_size': self.n_features,
            'hidden_size': 10,
            'output_size': self.n_actions
        })

        self.optimizer = torch.optim.Adam(self.pollicy_net.parameters(),
                                          lr=self.lr)
        # 不进行reduction操作，方便后面和v_t相乘
        self.loss_func = torch.nn.NLLLoss(reduction='none')

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter sowaka
            self.writer = SummaryWriter("logs/")
            self.writer.add_graph(self.pollicy_net, torch.ones((2, n_features)))
            self.writer.add_scalar('a', self.n_actions)

        # self.sess.run(tf.global_variables_initializer())

    def choose_action(self, observation):
        # 推断的时候是一个一个输入的，所以这里要加一个唯独
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        pro_weights = self.pollicy_net(observation)
        # 概率分布
        self.ep_as_prob.append(pro_weights[0].data)
        m = Categorical(pro_weights)
        # 变为类别分布，然后采样
        action = m.sample()
        return action.item()
        # return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = torch.FloatTensor(
            self._discount_and_norm_rewards())

        # return discounted_ep_rs_norm
        # 这里要梯度上升，最大化似然
        loss = torch.mean(-self.loss_func(torch.stack(self.ep_as_prob),
                                          torch.LongTensor(
                                              self.ep_as)) *
                          discounted_ep_rs_norm)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # train on episode

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        # 因为每次是收集的一个回合的数据，要重新处理下reward，以便满足折扣银子的要求
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            # 越近的reward的重要程度越高
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


if __name__ == '__main__':
    pg = PolicyGradient(
        n_actions=2,
        n_features=4,
        learning_rate=0.01,
        reward_decay=0.95,
        output_graph=True
    )
