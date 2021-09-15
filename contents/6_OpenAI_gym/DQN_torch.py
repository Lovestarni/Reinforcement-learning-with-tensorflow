'''
Dependencies:
    - torch
    - torchvision
    - gym
    - numpy
    - matplotlib
    - tqdm
'''
import numpy as np
import torch
from tensorboardX import SummaryWriter


# from torch.utils.tensorboard import SummaryWriter


class Net(torch.nn.Module):
    def __init__(self, configs) -> None:
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(configs['input_size'],
                                   configs['hidden_size'])
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1_ac = torch.nn.ReLU()
        self.out = torch.nn.Linear(configs['hidden_size'], configs[
            'output_size'])
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc1_ac(x)
        out = self.out(x)
        return out


class DeepQNetwork:
    def __init__(self,
                 n_actions=10,
                 n_features=50,
                 hidden_size=30,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=0.1,
                 output_graph=True, ) -> None:
        self.n_actions = n_actions
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.output_graph = output_graph

        # total learning step
        # learning step 是独立计数的，方便统计
        self.learn_step_counter = 0
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, (self.n_features * 2 + 2)))
        configs = {'input_size': self.n_features, 'hidden_size':
            self.hidden_size, 'output_size': self.n_actions}

        self.eval_net, self.target_net = Net(configs), Net(configs)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=self.lr)
        self.loss_func = torch.nn.MSELoss()

        if self.output_graph:
            self.writer = SummaryWriter('./logs')
            self.writer.add_graph(self.eval_net, torch.FloatTensor(np.zeros(
                self.n_features)))

    def choose_action(self, observation):
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)
        # input only one sample
        if np.random.uniform() < self.epsilon:  # greedy
            actions_value = self.eval_net.forward(observation)
            action = torch.max(actions_value, 1)[1].numpy()
            # torch.max会返回最大值和所在的索引
            # 返回的实在array中
            action = action[0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        # 循环覆盖保存
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            # update target net state
            # pytorch的网络模型更换操作很优雅
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # increasing epsilon, 开始阶段增大随机探索的概率，随着训练稳定，逐步减小随机性，保证稳定
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon \
                                                                < \
                                                                self.epsilon_max else self.epsilon_max

        # sample batch transitions
        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_features])
        b_a = torch.LongTensor(b_memory[:, self.n_features:self.n_features + 1])
        b_r = torch.FloatTensor(b_memory[:,
                                self.n_features + 1:self.n_features + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_features:])

        # 感觉传统的dqn，在开始阶段收集经验的时候效率会很低，因为都是没有学习的，
        # 是不是可以再最开始的batch进行改进，提高初始训练的效率
        # q_eval w.r.t the action in the experience
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # gather方法，可以根据index来收集列表中的数据
        q_next = self.target_net(b_s_).detach()
        # detach from the graph, avoid to backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        # torch的max都会返回最大值和位置，很方便
        loss = self.loss_func(q_eval, q_target)

        # add to the tensorboard
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # write to tensorboard
        # 实际中平均回报率会更为使用，cost和实际训练效果有时候在强化学习中没有什么关系
        # 因为拟合的q函数没有办法证明一定会收敛
        if self.output_graph:
            self.writer.add_scalar('loss', loss, self.learn_step_counter)


if __name__ == '__main__':
    dqn = DeepQNetwork()
