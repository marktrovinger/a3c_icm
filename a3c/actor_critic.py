import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, gamma=0.99, beta=0.01, hidden_units=256):

        self.conv1 = nn.Conv2d(input.shape, 32, kernel_size=(3,3), stride=2,
                padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
