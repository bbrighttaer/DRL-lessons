# Author: bbrighttaer
# Project: DRL-lessons
# Date: 1/16/20
# Time: 11:14 PM
# File: io.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import torch


def load_model(path, name, dvc='cpu'):
    """
    Loads the parameters of a model.

    :param path:
    :param name:
    :param dvc:
    :return: The saved state_dict.
    """
    dvc = torch.device(dvc)
    return torch.load(os.path.join(path, name), map_location=dvc)


def save_model(model, path, name):
    """
    Saves the model parameters.

    :param model:
    :param path:
    :param name:
    :return:
    """
    os.makedirs(path, exist_ok=True)
    file = os.path.join(path, name + ".mod")
    torch.save(model.state_dict(), file)
