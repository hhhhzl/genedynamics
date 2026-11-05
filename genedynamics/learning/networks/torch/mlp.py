# Adapted from zita-ch/supermimic_vibed @ 0525a23a2dcfa61f8543f98b84c075fe66d9d286.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.supermimic_vibed.
"""Dense networks shared by predictive models and controllers."""

from torch import nn


def mlp(
    inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False
) -> nn.Sequential:
    layers, d = ([], inp)
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


def run_mlp(net, values):
    return net(values)
