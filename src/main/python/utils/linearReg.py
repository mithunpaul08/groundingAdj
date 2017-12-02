#!/usr/bin/env python
from __future__ import print_function
from itertools import count
import torch.nn as nn
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
POLY_DEGREE = 1
##W_target = torch.randn(POLY_DEGREE, 1) * 5
#b_target = torch.randn(1) * 5
import torch.optim as optim
#
# def make_features(x):
#     """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
#     x = x.unsqueeze(1)
#     return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)
#

# def f(x):
#     """Approximated function."""
#     return x.mm(W_target) + b_target[0]


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x '.format(w)
    result += '{:+.2f}'.format(b[0])
    return result


def convert_variable(features, labels):
    """Builds a batch i.e. (x, f(x)) pair."""

    #the actual features and y comes here.
    # Toy data
    # x = np.random.randn(3, 3).astype('f')
    # x2=torch.from_numpy(x)
    # w = np.array([[1], [2], [3]],dtype="float32")
    # y = np.dot(x, w)
    # y2 = torch.from_numpy(y)

    #actual data


    x2 =torch.from_numpy(features)
    y2 = torch.from_numpy(labels)

    print("x2")
    print(x2)
    print("y2")
    print(y2)

    return Variable(x2), Variable(y2,requires_grad=False)

def runLR(features, y):
    featureShape=features.shape
    #print("featureShape")
    #print(featureShape)

    # create teh weight matrix. the dimensions must be transpose
    # of your features, since they are going to be dot producted


    fc = torch.nn.Linear(featureShape[1],1)
    #fc = torch.nn.Linear(3, 1)

    #print(fc)

    #y = Variable(torch.randn(N, D_out), requires_grad=False)

    #randX = np.random()


    for epoch in range(100):

        # Reset gradients
        fc.zero_grad()

        #np.random.shuffle(features)
        # Get data
        batch_x, batch_y = convert_variable(features, y)




        loss_fn = nn.MSELoss(size_average=True)
        optimizer = optim.SGD(fc.parameters(), lr=0.00001)


        #print(batch_x)
        #print(batch_y)
        #sys.exit(1)

        #multiply weight with input vector
        affine=fc(batch_x)
        #print(affine)
        #sys.exit(1)
        # for i in range(10):
        #     print("Predicted: {0}\tActual: {1}".format(affine[i], y[i]))
        #
        # sys.exit(1)
        loss = loss_fn(affine, batch_y)
        # Forward pass
        #output = nn.MSELoss(
        #loss = output.data[0]


        # Backward pass
        loss.backward()

        optimizer.step()

        # # Apply gradients
        # for param in fc.parameters():
        #     param.data -= 0.001 * param.grad.data

        # for param in fc.parameters():
        #     param.data.add_(-0.1 * param.grad.data)

        print("loss")
        print(loss)


        # # Stop criterion
        # if loss < 1e-3:
        #     break

        #print('Loss: {:.6f} after {} epochs'.format(loss.data, epoch))
    #print("weight:")
    print(fc.weight.data.view(-1))
   # print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    #print('==> Actual function:\t' + (W_target.view(-1), b_target))
