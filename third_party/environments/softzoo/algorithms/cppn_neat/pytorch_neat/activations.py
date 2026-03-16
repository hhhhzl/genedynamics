# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import torch
import torch.nn.functional as F


def sigmoid_activation(x):
    return torch.sigmoid(5 * x)


def tanh_activation(x):
    return torch.tanh(2.5 * x)


def sin_activation(x):
    return torch.sin(x)


def gauss_activation(x):
    return torch.exp(-5.0 * x**2)


def relu_activation(x):
    return F.relu(x)


def elu_activation(x):
    return F.elu(x)


def lelu_activation(x):
    return F.lelu(x)


def selu_activation(x):
    return F.selu(x)


def softplus_activation(x):
    return F.softplus(x)


def identity_activation(x):
    return x


def clamped_activation(x):
    return torch.clamp(x, min=-1., max=1.)


def inv_activation(x):
    try:
        out = 1.0 / x
    except ArithmeticError:
        return 0.0
    else:
        return out


def log_activation(x):
    return torch.log(torch.clamp(x, min=1e-7))


def exp_activation(x):
    return torch.exp(torch.clamp(x, min=-60., max=60.))


def abs_activation(x):
    return torch.abs(x)


def hat_activation(x):
    return torch.clamp(1 - torch.abs(x), min=0.0)


def square_activation(x):
    return x ** 2


def cube_activation(x):
    return x ** 3


str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'sin': sin_activation,
    'gauss': gauss_activation,
    'relu': relu_activation,
    'elu': elu_activation,
    'lelu': lelu_activation,
    'selu': selu_activation,
    'softplus': softplus_activation,
    'identity': identity_activation,
    'clamped': clamped_activation,
    'inv': inv_activation,
    'log': log_activation,
    'exp': exp_activation,
    'abs': abs_activation,
    'hat': hat_activation,
    'square': square_activation,
    'cube': cube_activation,
}
