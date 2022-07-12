# Define model
import torch.nn as nn

class FullConnectedNeuralNetwork(nn.Module):
    def __init__(self, withSoftmax: bool = False):
        super(FullConnectedNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        if(withSoftmax):
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.Softmax(),
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
            )
            
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits