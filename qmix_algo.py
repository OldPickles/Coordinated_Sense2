import time

import torch
import torch.nn as nn
import numpy as np
import random
import os

from Agent_net import Q_network_MLP
from mix_net_v0 import Qmix_network


# orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class QMixAlgo:
    def __init__(self, args):
        self.args = args
        self.env = args.env
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.observation_dim = args.observation_dim
        self.agent_net_hidden_dim = args.agent_net_hidden_dim

        self.state_dim = args.state_dim
        self.avail_actions_matrix_dim = args.avail_actions_matrix_dim
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim

        self.max_episodes = args.max_episodes
        self.lr = args.lr
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq
        self.use_lr_decay = args.use_lr_decay
        self.use_orthogonal_init = args.use_orthogonal_init
        self.use_RMS = args.use_RMS

        print("Agent Net Initializing...")
        time.sleep(1)
        self.eval_agent_net = Q_network_MLP(input_dim=self.observation_dim, hidden_dim=self.agent_net_hidden_dim, output_dim=self.avail_actions_matrix_dim).to(self.device)
        self.target_agent_net = Q_network_MLP(input_dim=self.observation_dim, hidden_dim=self.agent_net_hidden_dim,  output_dim=self.avail_actions_matrix_dim).to(self.device)
        self.target_agent_net.load_state_dict(self.eval_agent_net.state_dict())
        if self.use_orthogonal_init:
            orthogonal_init(self.eval_agent_net)
            orthogonal_init(self.target_agent_net)

        print("Qmix Net Initializing...")
        time.sleep(1)
        self.eval_mix_net = Qmix_network(qmix_hidden_dim=self.qmix_hidden_dim, hyper_hidden_dim=self.hyper_hidden_dim,
                                         env=self.env).to(self.device)
        self.target_mix_net = Qmix_network(qmix_hidden_dim=self.qmix_hidden_dim, hyper_hidden_dim=self.hyper_hidden_dim,
                                           env=self.env).to(self.device)
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        if self.use_orthogonal_init:
            orthogonal_init(self.eval_mix_net)
            orthogonal_init(self.target_mix_net)

        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_agent_net.parameters())
        if self.use_RMS:
            print("------optimizer: RMS prop------")
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam prop------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)

        self.train_times = 0
        pass

    def choose_action(self, observations, avail_actions, avail_actions_matrix, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [random.choice(avail_a) for avail_a in avail_actions]
            else:
                inputs = torch.tensor(observations, dtype=torch.float32).to(self.device)  # obs_n.shape=(N，obs_dim)
                q_value = self.eval_agent_net(inputs)

                avail_a_n = torch.tensor(avail_actions_matrix, dtype=torch.float32).to(self.device)  # avail_a_n.shape=(N, action_dim)
                avail_a_n = avail_a_n.reshape(q_value.shape)
                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                q_value = q_value.reshape((self.env.n_worker,) + self.env.avail_actions_matrix_shape).cpu()

                # choose action with highest Q-value
                a_n = []
                for q_value_i in q_value:
                    actions = np.where(q_value_i == torch.max(q_value_i))
                    a_n.append([actions[0][0], actions[1][0]])
            a_n = np.array(a_n)
            return a_n.reshape((self.env.n_worker,) + self.env.action_shape)

    def train(self, batch, total_episodes):
        """
                根据给出的经验:batch 训练模型
                :return:
                """
        self.train_times += 1

        # agent_net
        q_net_eval_input = torch.tensor(batch["last_observations"], requires_grad=True).float().to(self.device)
        q_values_eval = self.eval_agent_net.forward(q_net_eval_input).reshape((-1, self.env.n_worker,) + self.env.avail_actions_matrix_shape)
        q_net_target_input = torch.tensor(batch["next_observations"], requires_grad=True).float().to(self.device)
        q_values_target = self.target_agent_net.forward(q_net_target_input).reshape((-1, self.env.n_worker,) + self.env.avail_actions_matrix_shape)

        # 将q_values_eval中的q值,依据所选取的动作取出来对应q值,
        # shape: batch_size * a_agents * avail_actions_shape -> batch_size * a_agents
        actions = batch["actions"]
        actions_index = [(action[:,0] * self.env.avail_actions_matrix_shape[1] + action[:, 1]).tolist() for action in actions]
        actions_index = torch.tensor(actions_index, dtype=torch.int64).to(self.device)
        q_values_eval = q_values_eval.reshape((-1, self.env.n_worker, self.env.avail_actions_matrix_dim))
        q_values_eval = q_values_eval.gather(dim=-1, index=actions_index.unsqueeze(-1))
        q_values_eval = q_values_eval.squeeze()  # shape = (batch_size, a_agents)
        pass
        # 将q_values_target中的q值,依据下一步可执行的动作,选取其中最大的q值.
        # shape: batch_size * a_agents * avail_actions_shape -> batch_size * a_agents
        next_avail_actions_matrix = torch.tensor(batch['next_avail_actions_matrix'], dtype=torch.int64).to(self.device)
        q_values_target[next_avail_actions_matrix == 0] = -np.inf
        q_values_target = q_values_target.reshape((-1, self.env.n_worker, self.env.avail_actions_matrix_dim))
        q_values_target = q_values_target.max(dim=-1, keepdim=True)[0].squeeze()  # shape = batch_size * a_agents
        pass

        # qmix_net
        mix_net_eval_input = torch.tensor(batch['last_states'], requires_grad=True).float().to(self.device)
        mix_net_eval_input = mix_net_eval_input[:, :]  # 在一个批次里面,所有智能体的全局状态都是一样的.
        mix_net_target_input = torch.tensor(batch['next_states'], requires_grad=True).float().to(self.device)
        mix_net_target_input = mix_net_target_input[:, :]  # 在一个批次里面,所有智能体的全局状态都是一样的.

        q_total_eval = self.eval_mix_net.forward(q_values_eval, mix_net_eval_input)
        q_total_eval = q_total_eval.squeeze()  # shape = (batch_size,)
        q_total_target = self.target_mix_net.forward(q_values_target, mix_net_target_input)
        q_total_target = q_total_target.squeeze()  # shape = (batch_size,)

        # 计算奖励
        rewards = torch.tensor(batch['rewards']).float().to(self.device)
        rewards = rewards.squeeze()  # 使得rewards的shape为 (batch_size, n_agents)
        rewards = torch.sum(rewards, dim=-1, keepdim=True)
        rewards = rewards.squeeze()  # 使得rewards的shape为 (batch_size,)

        # 回合是否结束
        terminates = torch.tensor(batch['terminate'], ).float().to(self.device)
        terminates = terminates.squeeze()  # 使得terminates的shape为 (batch_size,)

        # 目标q值
        q_total_target = rewards + self.gamma * (1 - terminates) * q_total_target

        # 损失函数
        # torch.nn.MSELoss()  # 均方误差
        # loss = self.loss_func(q_total_eval, q_total_target)
        # 时序误差
        loss = (q_total_eval - q_total_target.detach()) ** 2

        # 梯度清零
        self.optimizer.zero_grad()

        # 反向传播
        loss.sum().backward()
        self.optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_episodes)

        if self.train_times % self.target_update_freq == 0:
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            self.target_agent_net.load_state_dict(self.eval_agent_net.state_dict())

        return loss.mean()


    def lr_decay(self, total_episodes):  # Learning rate Decay
        lr_now = self.lr * (1 - total_episodes / self.max_episodes)
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def save_model(self, save_path):
        agent_net_path = os.path.join(save_path, 'agent_net.pth')
        mix_net_path = os.path.join(save_path,'mix_net.pth')
        if not os.path.exists(save_path):
            print(f"{save_path} doesn't exist, creating it...")
            os.makedirs(save_path)
        torch.save(self.eval_mix_net.state_dict(), agent_net_path)
        torch.save(self.eval_agent_net.state_dict(), mix_net_path)
        print(f"agent_net.pth and mix_net.pth saved to {save_path}")
