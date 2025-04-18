import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Q_network_MLP(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=64, output_dim=None, use_orthogonal_init=True):
        super(Q_network_MLP, self).__init__()

        if input_dim is None or hidden_dim is None:
            raise ValueError("input_dim and hidden_dim must be specified")

        self.use_orthogonal_init = use_orthogonal_init

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        if self.use_orthogonal_init:
            # 判断是否对网络的权重使用正交初始化。
            print("------use orthogonal init in Q_network_MLP------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs, ):
        """
        When 'choose_action',
            inputs.shape(N,input_dim)
            Q.shape = (N, self.env.avail_actions_dim)
        When 'train',
            inputs.shape(bach_size,N,input_dim)
            Q.shape = (batch_size, N, self.env.avail_actions_dim)
        :param inputs:
        :return:
        """
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return Q
