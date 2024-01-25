import torch.nn as nn
import torch


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(4, 32) # 4 neuron in input layer, 32 neurons in 1st hidden layer 1
        self.hidden2 = nn.Linear(32, 64) # 64 neurons in 2nd hidden layer
        self.hidden3 = nn.Linear(64, 128) # 128 neurons in 3nd hidden layer
        self.output = nn.Linear(128, 3) # 3 neuron in output layer

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x