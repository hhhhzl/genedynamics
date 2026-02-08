import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

import diffuser.utils as utils

#-----------------------------------------------------------------------------#
#---------------------------------- modules ----------------------------------#
#-----------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

#-----------------------------------------------------------------------------#
#--------------------------------- attention ---------------------------------#
#-----------------------------------------------------------------------------#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        return self.to_out(out)

#-----------------------------------------------------------------------------#
#---------------------------------- sampling ---------------------------------#
#-----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Explanation: https://www.kirschstein.io/files/oxford-tdl.pdf
    Without s, the noise in the initial time steps is imperceptible, which slows down training.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

def apply_conditioning(x, conditions, action_dim, goal_dim=0, noise=False):
    '''
        x : tensor
            [ batch_size x horizon x (action_dim + obs_dim + goal_dim) ]
        conditions : dict
            { t: values }, where values is a batch_size x obs_dim tensor
        action_dim : int
            dimension of the action space
        goal_dim : int
            dimension of the goal space
        noise : bool
            Indicates if conditioning has to be applied to the added noise (i.e., set entries to zero)
    '''
    
    for t, val in conditions.items():
        if isinstance(t, str):     # unsafe sets
            continue
        else:
            x[:, t, action_dim:] = val.clone() if not noise else 0
    
    if goal_dim > 0:
        x[:, :, -goal_dim:] = conditions[0][:, -goal_dim:].unsqueeze(1).clone() if not noise else 0

    return x

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self, weights, action_dim):
        super().__init__()
        self.register_buffer('weights', weights)
        self.action_dim = action_dim

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        a0_loss = (loss[:, 0, :self.action_dim] / self.weights[0, :self.action_dim]).mean()
        return weighted_loss, {'diffusion_loss': weighted_loss, 'a0_loss': a0_loss}
    
class WeightedStateLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer('weights', weights)

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss, {'diffusion_loss': weighted_loss}
    
class DynamicsLoss(nn.Module):
    def __init__(self, weights, A_dyn, b_dyn):
        super().__init__()
        self.register_buffer('weights', weights)
        self.A_dynT = A_dyn.t()
        self.b_dynT = b_dyn.t()

    def forward(self, pred, targ):
        '''
            pred, targ : tensor
                [ batch_size x horizon x transition_dim ]
        '''
        pred = einops.rearrange(pred, 'b h d -> b (h d)') @ self.A_dynT 
        # pred = pred.contiguous().view(pred.shape[0], -1) @ self.A_dynT 
        targ = self.b_dynT
        loss = self._loss(pred, targ).mean()
        return loss, {'dyn_loss': loss}


class ValueLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, pred, targ):
        loss = self._loss(pred, targ).mean()

        if len(pred) > 1:
            corr = np.corrcoef(
                utils.to_np(pred).squeeze(),
                utils.to_np(targ).squeeze()
            )[0,1]
        else:
            corr = np.NaN
        
        info = {
            'mean_pred': pred.mean(), 'mean_targ': targ.mean(),
            'min_pred': pred.min(), 'min_targ': targ.min(),
            'max_pred': pred.max(), 'max_targ': targ.max(),
            'corr': corr,
        }
        info = {}

        return loss, info

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')
    
class WeightedStateL2(WeightedStateLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')
    
class WeightedDynamicsL2(DynamicsLoss):
    
        def _loss(self, pred, targ):
            return F.mse_loss(pred, targ, reduction='none')
            
class ValueL1(ValueLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class ValueL2(ValueLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')

class CrossEntropy(ValueLoss):

    def _loss(self, pred, target):
        pred = pred.reshape(pred.shape[0] * int(pred.shape[1]/2), 2)
        target = target.reshape(target.shape[0] * target.shape[1], ).type(torch.long)
        return F.cross_entropy(pred, target)

Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
    'state_l2': WeightedStateL2,
    'dynamics_l2': WeightedDynamicsL2,
    'value_l1': ValueL1,
    'value_l2': ValueL2,
    'cross_entropy': CrossEntropy
}

def get_loss_weights(action_weight, transition_dim, action_dim, horizon, discount=1.0):
    '''
        sets loss coefficients for trajectory

        action_weight   : float
            coefficient on first action loss
        discount   : float
            multiplies t^th timestep of trajectory loss by discount**t
        transition_dim : int
            dimension of the transition space
        action_dim : int
            dimension of the action space
        horizon : int
            length of trajectory
    '''

    dim_weights = torch.ones(transition_dim, dtype=torch.float32)

    ## decay loss with trajectory timestep: discount**t
    discounts = discount ** torch.arange(horizon, dtype=torch.float)
    discounts = discounts / discounts.mean()
    loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

    ## manually set a0 weight
    loss_weights[0, :action_dim] = action_weight
    return loss_weights
