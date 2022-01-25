import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_dims, gamma=0.99, beta=0.01, hidden_units=256):
        
        super(ActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=(3,3), stride=2,
                padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv_shape = self.calc_output_shape()

        self.gru = nn.GRUCell(self.conv_shape, 256)

