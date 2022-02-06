import numpy as np
import torch as T
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, n_actions, input_dims, gamma=0.99, beta=0.01, hidden_units=256):
        
        super(ActorCritic, self).__init__()
        self.gamma = gamma
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=(3,3), stride=2,
                padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=2, padding=1)

        self.conv_shape = self.calculate_conv_output(input_dims)

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

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample()
        log_probs =  dist.log_prob(action)

        return action.numpy()[0], v, log_probs, hidden
    
    def calcR(self, done, reward, values):
        values = T.cat(values).squeeze()
    
        # if we have a batch of states
        if len(values.size()) == 1:
            R = values[-1] * (1-int(done))
        # or if we have a single value
        elif len(values.size()) == 0:
            R = values * (1-int(done))

        # calculate our batch returns, reverse the list and convert to tensor
        
        batch_returns = []
        
        for r in reward[::-1]:
           R = r + self.gamma * R
           batch_returns.append(reward)
        batch_returns.reverse()
        batch_returns = T.tensor(batch_returns, 
                                dtype=T.float).reshape(values.size())
        return batch_returns

    def calc_cost(self, state_, hx, done, rewards, values, log_probs):
        # calc returns
        R = self.calcR(done, rewards, values)

        next_v = T.zeros(1,1) if done else self.forward(T.tensor([state_],
                                                        dtype=T.float), hx)[1]
        values.append(next_v.detach())
        values = T.cat(values).squeeze()

