import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, n, N):
        super(Net, self).__init__()
        self.N = N
        self.n = n
        self.hidden1 = nn.Linear(N, 32) # N neuron in input layer, 32 neurons in 1st hidden layer 1
        self.hidden2 = nn.Linear(32, 64) # 64 neurons in 2nd hidden layer
        self.hidden3 = nn.Linear(64, 128) # 128 neurons in 3nd hidden layer
        self.output = nn.Linear(128, n+1) # n + 1 (n + alpha) neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x

class NetAlpha(nn.Module):
    def __init__(self, n, N):
        super(Net, self).__init__()
        self.N = N
        self.n = n
        self.hidden1 = nn.Linear(N, 32) # N neuron in input layer, 32 neurons in 1st hidden layer 1
        self.hidden2 = nn.Linear(32, 64) # 64 neurons in 2nd hidden layer
        self.hidden3 = nn.Linear(64, 128) # 128 neurons in 3nd hidden layer
        self.output = nn.Linear(128, 1) # alpha, 1 neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x

