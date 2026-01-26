import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange
import pdb

from enerdynamics.solvers.single.dpcc import diffuser_utils as utils

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
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
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
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        mean = x.mean(dim=-1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = einops.rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(dots.dtype).max
            mask = einops.repeat(mask, "b j -> b h i j", h=h, i=dots.shape[-2])
            dots.masked_fill_(~mask, max_neg_value)

        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: einops.rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = einops.rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(dots.dtype).max
            mask = einops.repeat(mask, "b j -> b h i j", h=h, i=dots.shape[-2])
            dots.masked_fill_(~mask, max_neg_value)

        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = einops.rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        context_dim=None,
        dropout=0.0,
        use_checkpoint=False,
        ff_bias=True,
    ):
        super().__init__()
        self.attn1 = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.attn2 = CrossAttention(
            dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
        )
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        self.use_checkpoint = use_checkpoint

    def forward(self, x, cond=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, cond)
        return self._forward(x, cond)

    def _forward(self, x, cond=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond) + x
        x = self.ff(self.norm3(x)) + x
        return x

    def norm1(self, x):
        return F.layer_norm(x, x.shape[-1:])

    def norm2(self, x):
        return F.layer_norm(x, x.shape[-1:])

    def norm3(self, x):
        return F.layer_norm(x, x.shape[-1:])


#-----------------------------------------------------------------------------#
#--------------------------- diffusion helpers ------------------------------#
#-----------------------------------------------------------------------------#


class Losses:
    def __getitem__(self, key):
        return getattr(self, key)

    def l1(self, weights, action_dim):
        def _loss(pred, targ):
            loss = (pred - targ).abs()
            loss = loss * weights[None, :, :].to(pred)
            return loss.mean(dim=(1, 2))

        return _loss

    def l2(self, weights, action_dim):
        def _loss(pred, targ):
            loss = (pred - targ) ** 2
            loss = loss * weights[None, :, :].to(pred)
            return loss.mean(dim=(1, 2))

        return _loss

    def huber(self, weights, action_dim):
        def _loss(pred, targ):
            loss = F.smooth_l1_loss(pred, targ, reduction="none")
            loss = loss * weights[None, :, :].to(pred)
            return loss.mean(dim=(1, 2))

        return _loss

    def elbo(self, weights, action_dim):
        def _loss(x_start, pred_noise, noise, t):
            loss = (pred_noise - noise) ** 2
            loss = utils.apply_dict(lambda w: w[None, :, :], weights)
            loss = utils.apply_dict(lambda w: w.to(loss), loss)
            loss = loss["observations"] * loss["actions"]
            loss = utils.apply_dict(lambda w: w.to(loss.device), loss)
            loss = utils.apply_dict(lambda w: w * loss, loss)
            loss = utils.apply_dict(lambda w: w.sum(dim=-1), loss)
            return loss.mean(dim=1)

        return _loss

    def noise_pred(self, weights, action_dim):
        def _loss(x_start, pred_noise, noise, t):
            loss = (pred_noise - noise) ** 2
            return loss.mean(dim=(1, 2))

        return _loss


Losses = Losses()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def extract(a, t, x_shape):
    """
    extract values from a for batch of indices t.
    a: (T)
    t: (B)
    output: (B, 1, 1, 1)
    """
    bs = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(bs, *((1,) * (len(x_shape) - 1)))


def apply_conditioning(x, cond, action_dim, transition_dim):
    x = x.clone()
    for t, val in cond.items():
        x[:, t, action_dim:transition_dim] = val
    return x


#-----------------------------------------------------------------------------#
#--------------------------------- utilities ---------------------------------#
#-----------------------------------------------------------------------------#


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def swish(x):
    return x * torch.sigmoid(x)


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d
