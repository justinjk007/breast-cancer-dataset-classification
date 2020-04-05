import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, i, h1, h2, o):
        super(NeuralNetwork, self).__init__()

        # Inputs to hidden layer linear transformation
        self.layer1 = nn.Linear(i, h1)
        # Hidden layer 1 to HL2 linear transformation
        self.layer2 = nn.Linear(h1, h2)
        # HL2 to output linear transformation
        self.layer3 = nn.Linear(h2, o)

        # Define relu activation and LogSoftmax output
        self.relu = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # HL1 with relu activation
        out = self.relu(self.layer1(x))
        # HL2 with relu activation
        out = self.relu(self.layer2(out))
        # Output layer with LogSoftmax activation
        out = self.LogSoftmax(self.layer3(out))
        return out
