"""
layers.py — STGCN Building Blocks
===================================
Implements the core layers of the Spatio-Temporal Graph Convolutional Network
based on the paper: "Spatio-Temporal Graph Convolutional Networks: A Deep
Learning Framework for Traffic Forecasting" (Yu et al., 2018).

Architecture:
    STConvBlock = Temporal Gate Conv → Chebyshev Spatial Conv → Temporal Gate Conv
    (Sandwich structure: T-S-T)

Layers:
    TemporalConv:   Gated 1D causal convolution over time dimension
    ChebConv:       Chebyshev spectral graph convolution (spatial)
    STConvBlock:    Full spatio-temporal block
    OutputLayer:    Final FC output
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal Convolution (Gated)
# ---------------------------------------------------------------------------

class TemporalConv(nn.Module):
    """
    Gated Temporal Convolutional Layer.

    Applies a 1D causal convolution across the time dimension with a
    gating mechanism (GLU — Gated Linear Unit):
        output = tanh(conv_A(x)) ⊙ sigmoid(conv_B(x))

    Parameters
    ----------
    in_channels : int  — input feature channels
    out_channels : int — output feature channels
    kernel_size : int  — temporal kernel size (default 3)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # GLU requires 2 * out_channels from conv (split into gate + value)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=(kernel_size, 1),   # only convolve over time
            padding=(kernel_size - 1, 0),   # causal padding (pad left)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_channels, time, num_nodes)

        Returns
        -------
        out : torch.Tensor, shape (batch, out_channels, time, num_nodes)
        """
        # Apply conv → (batch, 2*out, time + pad, nodes)
        out = self.conv(x)

        # Remove extra padding (causal: only left-pad, trim right excess)
        # After padding (kernel_size-1) on both sides, trim the excess on the right
        out = out[:, :, : x.shape[2], :]   # keep only original time length

        # Split into value and gate
        value, gate = out.chunk(2, dim=1)  # each: (batch, out_channels, time, nodes)

        # Gated activation
        out = torch.tanh(value) * torch.sigmoid(gate)
        out = self.bn(out)
        return out


# ---------------------------------------------------------------------------
# Chebyshev Spectral Graph Convolution (Spatial)
# ---------------------------------------------------------------------------

class ChebConv(nn.Module):
    """
    Chebyshev Spectral Graph Convolution Layer.

    Approximates spectral graph convolution using Chebyshev polynomials
    T_0(L), T_1(L), ..., T_{Ks-1}(L) pre-computed from the graph Laplacian.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    Ks : int — Chebyshev polynomial order
    cheb_polys : list of np.ndarray, each (N, N) — pre-computed T_k matrices
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        Ks: int,
        cheb_polys: List[np.ndarray],
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Ks = Ks

        # Register Chebyshev matrices as non-trainable buffers
        for k, T_k in enumerate(cheb_polys):
            self.register_buffer(
                f"T_{k}",
                torch.FloatTensor(T_k)  # (N, N)
            )

        # Learnable weight matrix: maps Ks * in_channels → out_channels
        self.weight = nn.Parameter(
            torch.FloatTensor(Ks * in_channels, out_channels)
        )
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_channels, time, num_nodes)

        Returns
        -------
        out : torch.Tensor, shape (batch, out_channels, time, num_nodes)
        """
        batch, C_in, T, N = x.shape

        # Reshape: (batch*T, N, C_in)
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(batch * T, N, C_in)

        # Apply each Chebyshev polynomial: T_k @ x
        cheb_outputs = []
        for k in range(self.Ks):
            T_k = getattr(self, f"T_{k}")  # (N, N)
            # (N, N) @ (batch*T, N, C_in) → broadcast: (batch*T, N, C_in)
            out_k = torch.einsum("nm,bmc->bnc", T_k, x_reshaped)  # (batch*T, N, C_in)
            cheb_outputs.append(out_k)

        # Concatenate along channel dim: (batch*T, N, Ks*C_in)
        cheb_cat = torch.cat(cheb_outputs, dim=-1)

        # Linear projection: (batch*T, N, Ks*C_in) @ (Ks*C_in, C_out) → (batch*T, N, C_out)
        out = cheb_cat @ self.weight + self.bias  # (batch*T, N, C_out)

        # Reshape back: (batch, C_out, T, N)
        out = out.view(batch, T, N, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return F.relu(out)


# ---------------------------------------------------------------------------
# Spatio-Temporal Convolutional Block (T-S-T Sandwich)
# ---------------------------------------------------------------------------

class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block.

    Architecture:
        Input → TemporalConv → ReLU → ChebConv → LayerNorm → TemporalConv → Dropout

    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    out_channels : int
    Ks : int
    cheb_polys : list of (N, N) ndarray
    kernel_size : int — temporal kernel size
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        Ks: int,
        cheb_polys: List[np.ndarray],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # T-S-T: temporal → spatial → temporal
        self.temp_conv1 = TemporalConv(in_channels, hidden_channels, kernel_size)
        self.cheb_conv = ChebConv(hidden_channels, hidden_channels, Ks, cheb_polys)
        self.temp_conv2 = TemporalConv(hidden_channels, out_channels, kernel_size)

        # Layer norm over node and channel dims
        self.layer_norm = nn.LayerNorm([hidden_channels])
        self.dropout = nn.Dropout(dropout)

        # Residual projection if dims differ
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, in_channels, time, num_nodes)

        Returns
        -------
        out : torch.Tensor, shape (batch, out_channels, time, num_nodes)
        """
        residual = self.residual(x)

        # Temporal conv 1
        out = self.temp_conv1(x)  # (batch, hidden, time, nodes)

        # Spatial conv (Chebyshev)
        out = self.cheb_conv(out)  # (batch, hidden, time, nodes)

        # Layer norm over feature dim
        # Permute to (batch, time, nodes, hidden) → norm → permute back
        out = out.permute(0, 2, 3, 1)
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2)

        # Temporal conv 2
        out = self.temp_conv2(out)  # (batch, out_channels, time, nodes)

        out = self.dropout(out)
        out = out + residual
        return F.relu(out)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from preprocessing.graph_builder import build_graph
    import yaml

    logging.basicConfig(level=logging.DEBUG)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    graph = build_graph(cfg)
    Ks = cfg["stgcn"]["Ks"]
    cheb = graph["cheb_polys"]

    batch, T, N, C = 4, 12, 4, 1

    # Test TemporalConv
    x = torch.randn(batch, C, T, N)
    tc = TemporalConv(C, 16)
    out = tc(x)
    print(f"TemporalConv: {x.shape} → {out.shape}")
    assert out.shape == (batch, 16, T, N)

    # Test ChebConv
    x2 = torch.randn(batch, 16, T, N)
    cc = ChebConv(16, 64, Ks, cheb)
    out2 = cc(x2)
    print(f"ChebConv: {x2.shape} → {out2.shape}")
    assert out2.shape == (batch, 64, T, N)

    # Test STConvBlock
    x3 = torch.randn(batch, C, T, N)
    block = STConvBlock(C, 16, 64, Ks, cheb)
    out3 = block(x3)
    print(f"STConvBlock: {x3.shape} → {out3.shape}")
    print("All layer tests passed ✓")
