import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ICM(nn.Module):
    def __init__(self, input_dims, lambda_ = 0.1, beta = 0.2) -> None:
        super(ICM, self).__init__()

        # we use the same arch as actor-critic for the forward
        # model
        self.conv1 = nn.Conv2d(input_dims[0], 32, 3, stride=2,
                padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        def forward(self, state):
            pass

        def loss(self, state, state_):
            pass