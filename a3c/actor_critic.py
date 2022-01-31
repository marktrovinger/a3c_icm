import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, n_actions, input_dims, gamma=0.99, beta=0.01, hidden_units=256):
        
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=(3,3), stride=2,
                padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv_shape = self.calc_output_shape()

        self.gru = nn.GRUCell(self.conv_shape, 256)
        
        # pi is our policy layer, as opposed to v, our value network
        self.pi = nn.Linear(256, n_actions)
        self.v = nn.Linear(256, 1)


def calculate_conv_output(self, input_dims):
    state = T.zeros(1, *input_dims)
    dimensions = self.conv1(state)
    dimensions = self.conv2(dimensions)
    dimensions = self.conv3(dimensions)
    dimensions = self.conv4(dimensions)
    return int(np.product(dimensions.size()))

def forward(self, state, hidden_state):
    conv = F.elu(self.conv1(state))
    conv = F.elu(self.conv2(conv))
    conv = F.elu(self.conv3(conv))
    conv = F.elu(self.conv4(conv))

    conv_state = conv.view((conv.size()[0], -1))

    # handle our gated recurrent network
    hidden = self.gru(conv_state, (hidden_state))

    # send the output to our policy and value networks
    pi = self.pi(hidden)
    v = self.v(hidden)

