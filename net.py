import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Net, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden1 = nn.Linear(dim_in, 32) # N neuron in input layer, 32 neurons in 1st hidden layer 1
        self.hidden2 = nn.Linear(32, 64) # 64 neurons in 2nd hidden layer
        self.hidden3 = nn.Linear(64, 128) # 128 neurons in 3nd hidden layer
        self.output = nn.Linear(128, dim_out) # n + 1 (n + alpha) neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x

class NetAlpha(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NetAlpha, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden1 = nn.Linear(dim_in, 32) # N neuron in input layer, 32 neurons in 1st hidden layer 1
        self.hidden2 = nn.Linear(32, 64) # 64 neurons in 2nd hidden layer
        self.hidden3 = nn.Linear(64, 128) # 128 neurons in 3nd hidden layer
        self.output = nn.Linear(128, dim_out) # alpha, 1 neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x

class NetQ(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NetQ, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.hidden1 = nn.Linear(dim_in, 32) # N neuron in input layer, 32 neurons in 1st hidden layer 1
        self.hidden2 = nn.Linear(32, 64) # 64 neurons in 2nd hidden layer
        self.hidden3 = nn.Linear(64, 128) # 128 neurons in 3nd hidden layer
        self.hidden4 = nn.Linear(128, 256) # 128 neurons in 3nd hidden layer
        self.output = nn.Linear(256, dim_out) # alpha, 1 neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = self.output(x)
        return x

