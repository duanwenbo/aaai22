"""
# time: 07/09/2021
# update: /
# author: Bobby
Basic networks in Critic Actor framework.
"""
import torch.nn as nn
import torch.nn.functional as F


class Actor_net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))  # for probability
        return x


class Critic_net(nn.Module):
    def __init__(self, input, hidden, output):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x
