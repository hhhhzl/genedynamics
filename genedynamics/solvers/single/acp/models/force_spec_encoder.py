# Derived from Yifan Hou ACP (MIT), commit 74d68eca3bfb5cb59da131c6351d9e1c9b129ff6.
# Copyright (c) 2024 Yifan Hou@Stanford. See LICENSE and SOURCE.json.
import torch
import torch.nn as nn
from scipy import signal
import numpy as np


def convert_to_spec(data, fmin=0, fmax=200, sample_rate=7000, pad_spec=False):
    specs = []
    for i in range(data.shape[-1]):
        f, t, Sxx = signal.spectrogram(
            data[..., i].detach().cpu().numpy(),
            fs=sample_rate,
            nperseg=512,
            noverlap=512 // 4,
            nfft=1024,
        )
        freq_slice = np.where((f >= fmin) & (f <= fmax))
        # keep only frequencies of interest
        f = f[freq_slice]
        Sxx = Sxx[:, freq_slice, :]

        if pad_spec:
            padded_values = Sxx[:, :, :, 0]
            Sxx = np.concatenate([padded_values[:, :, :, None], Sxx], axis=-1)
        specs.append(Sxx)
    specs = np.concatenate(specs, axis=1)
    specs = torch.tensor(specs).float().to(data.device)  # (B, 6, 30, 17/18)
    return specs


class CoordConv(nn.Module):
    """Add coordinates in [0,1] to an image, like CoordConv paper."""

    def forward(self, x):
        # needs N,C,H,W inputs
        assert x.ndim == 4
        h, w = x.shape[2:]
        ones_h = x.new_ones((h, 1))
        type_dev = dict(dtype=x.dtype, device=x.device)
        lin_h = torch.linspace(-1, 1, h, **type_dev)[:, None]
        ones_w = x.new_ones((1, w))
        lin_w = torch.linspace(-1, 1, w, **type_dev)[None, :]
        new_maps_2d = torch.stack((lin_h * ones_w, lin_w * ones_h), dim=0)
        new_maps_4d = new_maps_2d[None]
        assert new_maps_4d.shape == (1, 2, h, w), (x.shape, new_maps_4d.shape)
        batch_size = x.size(0)
        new_maps_4d_batch = new_maps_4d.repeat(batch_size, 1, 1, 1)
        result = torch.cat((x, new_maps_4d_batch), dim=1)
        return result


class ForceSpecEncoder(nn.Module):
    def __init__(self, model, norm_spec):
        super().__init__()
        # modify model to take in 8 channels
        model[0] = nn.Conv2d(
            8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.model = model
        self.coord_conv = CoordConv()  # similar as positional encoding
        self.norm_spec = norm_spec

    def forward(self, spec):
        EPS = 1e-8
        # spec: B x C(6) x F x T
        log_spec = torch.log(spec + EPS)
        x = log_spec
        if self.norm_spec.is_norm:
            x = (x - self.norm_spec.min) / (self.norm_spec.max - self.norm_spec.min)
            x = x * 2 - 1

        x = self.coord_conv(x)
        # x: B x C(8) x F x T
        x = self.model(x)
        return x


class ForceSpecTransformer(nn.Module):
    def __init__(self, model, norm_spec):
        super().__init__()
        self.model = model
        self.norm_spec = norm_spec

    def forward(self, spec):
        EPS = 1e-8
        # spec: B x C(6) x F x T
        log_spec = torch.log(spec + EPS)
        x = log_spec
        if self.norm_spec.is_norm:
            x = (x - self.norm_spec.min) / (self.norm_spec.max - self.norm_spec.min)
            x = x * 2 - 1

        x = self.model(x)
        return x
