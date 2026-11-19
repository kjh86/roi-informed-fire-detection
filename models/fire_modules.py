# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
# Fixed-Edge Prior Fusion (FEPF) and Multi-Dilation Depthwise Mixer (MDDM)
# for "ROI-Based Visual Characteristic Analysis and Feature Enhancement for
# Fire and Smoke Detection across Multiple Data Distributions" (Fire, MDPI).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("FEPF", "MDDM")


class FEPF(nn.Module):
    """Fixed-Edge Prior Fusion (Sobel) - depthwise Sobel X/Y, edge magnitude, gated fusion."""

    def __init__(self, c1: int, c2: int | None = None):
        super().__init__()
        c = c2 if c2 is not None else c1
        self.c = c
        self.sobel_x = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.sobel_y = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        with torch.no_grad():
            self.sobel_x.weight.zero_()
            self.sobel_y.weight.zero_()
            for i in range(c):
                self.sobel_x.weight[i, 0] = kx
                self.sobel_y.weight[i, 0] = ky
        for p in self.sobel_x.parameters():
            p.requires_grad = False
        for p in self.sobel_y.parameters():
            p.requires_grad = False
        self.align = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(8, c // 16), 1),
            nn.SiLU(),
            nn.Conv2d(max(8, c // 16), c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ex = self.sobel_x(x)
        ey = self.sobel_y(x)
        e = torch.sqrt(ex * ex + ey * ey + 1e-6)
        e = self.align(e)
        g = self.gate(x)
        return x + e * g


class MDDM(nn.Module):
    """Multi-Dilation Depthwise Mixer - parallel depthwise dilated conv branches with softmax gating."""

    def __init__(self, c1: int, c2: int | None = None, dilations: tuple[int, ...] = (1, 2, 3), k: int = 3):
        super().__init__()
        c = c2 if c2 is not None else c1
        self.c = c
        self.proj = nn.Conv2d(c1, c, 1, bias=False) if c1 != c else nn.Identity()
        self.dilations = list(dilations)
        self.branches = nn.ModuleList([
            nn.Conv2d(c, c, k, stride=1, padding=d * (k // 2), dilation=d, groups=c, bias=False)
            for d in self.dilations
        ])
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(8, c // 16), 1),
            nn.SiLU(),
            nn.Conv2d(max(8, c // 16), len(self.dilations), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.proj(x)  # project to c when c1 != c
        ws = self.gate(x_in).flatten(1)
        ws = F.softmax(ws, dim=1)
        outs = []
        for i, conv in enumerate(self.branches):
            outs.append(conv(x_in) * ws[:, i].view(-1, 1, 1, 1))
        y = torch.stack(outs, dim=0).sum(dim=0)
        y = self.act(self.bn(self.pw(y)))
        return x_in + y
