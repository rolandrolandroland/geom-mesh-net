"""Throughput of the three model shapes the roadmap proposes, on this machine.

Synthetic tensors of the right shapes: nothing is fitted to data and nothing is
scored. The point is a compute budget for Stages 2, 3 and 5 of
``experiments/reconstruction/ROADMAP.md`` (section 4.4).

- a Fourier-feature MLP trained with BCE (Stage 2);
- a SIREN trained with BCE plus a Laplacian residual from autograd (Stage 5);
- a 3D U-Net over voxelised observations with an implicit decoder queried at
  continuous points (Stage 3).

``F.grid_sample`` has no 3D backward on MPS in torch 2.10, so the decoder's
feature lookup is written with ``gather``.

Usage
-----
    python -m experiments.reconstruction.pilot.timing
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

# This environment links two OpenMP runtimes; the neural-field scripts set the same flag.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from geom_mesh_net import paths


class FourierMLP(nn.Module):
    def __init__(self, n_freq=256, sigma=4.0, width=256, depth=4):
        super().__init__()
        self.register_buffer("B", torch.randn(n_freq, 3) * sigma)
        layers, d = [], 2 * n_freq
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        self.net = nn.Sequential(*layers, nn.Linear(d, 1))

    def forward(self, x):
        z = 2 * torch.pi * x @ self.B.T
        return self.net(torch.cat([z.sin(), z.cos()], dim=-1))


class Siren(nn.Module):
    def __init__(self, width=256, depth=4, w0=30.0):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3 if i == 0 else width, width) for i in range(depth)])
        self.out = nn.Linear(width, 1)
        self.w0 = w0

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.sin((self.w0 if i == 0 else 1.0) * layer(x))
        return self.out(x)


def trilinear(a, q):
    """Trilinear lookup of a (B, C, D, H, W) grid at q (B, N, 3) in [0, 1]; returns (B, N, C)."""
    B, C, D, H, W = a.shape
    size = torch.tensor([D, H, W], device=q.device, dtype=q.dtype)
    pos = q * (size - 1)
    i0 = pos.floor().clamp(min=torch.zeros(3, device=q.device), max=size - 2).long()
    t = pos - i0
    flat = a.reshape(B, C, -1)
    out = 0
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                idx = (i0[..., 0] + dz) * H * W + (i0[..., 1] + dy) * W + (i0[..., 2] + dx)
                w = ((t[..., 0] if dz else 1 - t[..., 0]) * (t[..., 1] if dy else 1 - t[..., 1])
                     * (t[..., 2] if dx else 1 - t[..., 2]))
                out = out + flat.gather(2, idx[:, None, :].expand(B, C, -1)) * w[:, None, :]
    return out.transpose(1, 2)


class ConvFieldNet(nn.Module):
    """Voxelised observations -> 3D U-Net feature grid -> MLP decoder at query points."""

    def __init__(self, c=(16, 32, 64), feat=32):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.GELU(),
                                 nn.Conv3d(o, o, 3, padding=1), nn.GELU())
        self.e1, self.e2, self.e3 = block(2, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.u2, self.u1 = block(c[2] + c[1], c[1]), block(c[1] + c[0], feat)
        self.dec = nn.Sequential(nn.Linear(feat + 3, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                 nn.Linear(128, 1))

    def forward(self, grid, q):
        a = self.e1(grid)
        b = self.e2(F.max_pool3d(a, 2))
        c = self.e3(F.max_pool3d(b, 2))
        b = self.u2(torch.cat([F.interpolate(c, scale_factor=2, mode="trilinear"), b], 1))
        a = self.u1(torch.cat([F.interpolate(b, scale_factor=2, mode="trilinear"), a], 1))
        return self.dec(torch.cat([trilinear(a, q), q], -1))


def time_steps(device, step, n, warm=5):
    for _ in range(warm):
        step()
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(n):
        step()
    if device.type == "mps":
        torch.mps.synchronize()
    return (time.time() - t0) / n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=paths.RECONSTRUCTION_DIR / "pilot" / "results" / "timing.json")
    args = parser.parse_args()

    devices = ([torch.device("mps")] if torch.backends.mps.is_available() else []) + [torch.device("cpu")]
    rows = []
    for device in devices:
        torch.manual_seed(0)
        bce = nn.BCEWithLogitsLoss()
        x = torch.rand(32768, 3, device=device)
        y = (torch.rand(32768, 1, device=device) < 0.1).float()

        ff = FourierMLP().to(device)
        opt = torch.optim.Adam(ff.parameters(), 1e-3)

        def ff_step():
            opt.zero_grad()
            bce(ff(x), y).backward()
            opt.step()
        rows.append(("Fourier-feature MLP, BCE, batch 32,768", device.type, time_steps(device, ff_step, 30)))

        siren = Siren().to(device)
        opt2 = torch.optim.Adam(siren.parameters(), 1e-4)
        xc = torch.rand(8192, 3, device=device)

        def pinn_step():
            opt2.zero_grad()
            xr = xc.clone().requires_grad_(True)
            g = torch.autograd.grad(siren(xr).sum(), xr, create_graph=True)[0]
            lap = sum(torch.autograd.grad(g[:, i].sum(), xr, create_graph=True)[0][:, i] for i in range(3))
            (bce(siren(x), y) + lap.pow(2).mean()).backward()
            opt2.step()
        rows.append(("SIREN, BCE 32,768 + Laplacian residual on 8,192 points", device.type,
                     time_steps(device, pinn_step, 30 if device.type == "mps" else 10)))

        if device.type == "mps":
            net = ConvFieldNet().to(device)
            opt3 = torch.optim.Adam(net.parameters(), 1e-3)
            grid = torch.rand(4, 2, 48, 48, 48, device=device)
            q = torch.rand(4, 16384, 3, device=device)
            yq = (torch.rand(4, 16384, 1, device=device) < 0.1).float()

            def conv_step():
                opt3.zero_grad()
                bce(net(grid, q), yq).backward()
                opt3.step()
            rows.append((f"3D U-Net ({sum(p.numel() for p in net.parameters()):,} params) + implicit "
                         "decoder, 4 x 48^3 crops, 16,384 queries", device.type, time_steps(device, conv_step, 10)))

    for name, dev, seconds in rows:
        print(f"{name:<86} {dev:>4}: {1000 * seconds:8.1f} ms/step")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        machine=platform.machine(), torch=torch.__version__,
        steps=[dict(model=n, device=d, ms_per_step=round(1000 * s, 1)) for n, d, s in rows]), indent=1) + "\n")
    print(f"written {args.output}")


if __name__ == "__main__":
    main()
