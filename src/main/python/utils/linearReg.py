#!/usr/bin/env python
from __future__ import print_function
from itertools import count
import torch.nn as nn
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable

POLY_DEGREE = 1
W_target = torch.randn(POLY_DEGREE, 1) * 5
b_target = torch.randn(1) * 5


def make_features(x):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)


def f(x):
    """Approximated function."""
    return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x^{} '.format(w, len(W) - i)
    result += '{:+.2f}'.format(b[0])
    return result


def get_batch(features, labels,batch_size=10):
    """Builds a batch i.e. (x, f(x)) pair."""

    #the actual features and y comes here.
    # random = torch.randn(batch_size)

    x = torch.from_numpy(features)
    y = torch.from_numpy(labels)
    return Variable(x), Variable(y)

def runLR(features, y):

    # Define model-i.e the inputs are just sizes or dimensions of the weight matrix
    fc = torch.nn.Linear(W_target.size(0), 1)

    for batch_idx in count(1):
        # Get data
        batch_x, batch_y = get_batch(features, y)

        # Reset gradients
        fc.zero_grad()


        loss = nn.MSELoss()
        output = loss(fc(batch_x), batch_y)
        # Forward pass
        #output = nn.MSELoss(
        loss = output.data[0]

        # Backward pass
        output.backward()

        # Apply gradients
        for param in fc.parameters():
            param.data.add_(-0.1 * param.grad.data)

        # Stop criterion
        if loss < 1e-3:
            break

    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
