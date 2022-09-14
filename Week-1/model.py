from typing import List, Tuple
import torch.nn as nn
from torch import optim
import numpy as np
import pandas as pd
import torch
import scipy
import collections
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from data import sample_data
from torch.autograd import Variable

def construct_mlp(parameters: List[Tuple[int, int]]) -> nn.Module:
    """Constructs a multi-layer perception (mlp) given
    the layer input and output sizes described as tuples within the list.

    Args:
        parameters (List[Tuple[int, int]]): First element of the tuple
        describes the input size, second describes the output size of a given layer.
        First element of the list is the first layer.

    Returns:
        nn.Module: Multi-layer perceptron
    """
    # Get number of layers
    num_layers = len(parameters)

    # Creates layers in an order Linear, Tanh, Linear, Tanh,.. and so on.. using list comprehension
    layers = [
        [nn.Linear(parameters[i][0], parameters[i][1]), nn.Tanh()]
        for i in range(num_layers - 1)
    ]
    layers = [layer for sublist in layers for layer in sublist]

    # Append last layer with sigmoid
    layers.append(
        nn.Linear(parameters[num_layers - 1][0], parameters[num_layers - 1][1])
    )
    layers.append(nn.Sigmoid())

    # Convert into dictionary
    layers = {"{}".format(i): item for i, item in enumerate(layers)}

    # Convert into OrderedDict
    layers = collections.OrderedDict(sorted(layers.items(), key=lambda t: t[0]))

    # Pass the ordered dictionary into nn.Sequential to define a model and return
    return nn.Sequential(layers)


def decision_boundary(model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """Exctracts the decision boundary constructed by a model

    Args:
        model (nn.Module): Has to be a model with last layer being a linear layer
        with the output of size 2 and intercept

    Returns:
        Tuple[np.ndarray, np.ndarray]: _description_
    """
    model_param = list(model.parameters())
    if torch.cuda.is_available():
        # Get last two hidden units weights, last hidden layer
        a = model_param[-2].data.cpu().numpy().ravel()[0]
        b = model_param[-2].data.cpu().numpy().ravel()[1]
        # Get bias for last hidden layer, just one element, since output is 1
        c = model_param[-1].data.cpu().numpy()[0]
    else:
        a = model_param[-2].data.numpy().ravel()[0]
        b = model_param[-2].data.numpy().ravel()[1]
        c = model_param[-1].data.numpy()[0]

    x_boundary = np.linspace(-1, 1, 100)
    y_boundary = (scipy.special.logit(0.5) - c - a * x_boundary) / b

    return x_boundary, y_boundary


def one_pass(model, loss_criterion, optimizer, df, N):
    """
    Does one optimisation step with batch size N
    """
    inp, label = sample_data(df, N = N) # Samples N number of data points from our data

    inputs = Variable(inp) # Passes them into a pytorch Variable
    target = Variable(label)

    optimizer.zero_grad() # Zero pytorch gradients
    output = model(inputs).squeeze() # Process the data through model

    loss = loss_criterion(output, target.squeeze()) # Compute the loss

    loss.backward() # Backpropogate the loss to accumulate gradients

    optimizer.step() # Update the weights using the accumulated gradients
    
    # Just return loss for plotting
    if torch.cuda.is_available():
        return loss.data.cpu().numpy()
    else:
        return loss.data.numpy()
