"""
stgcn_model.py — Full STGCN Architecture
==========================================
Spatio-Temporal Graph Convolutional Network for traffic forecasting.

Architecture overview:
    Input (batch, seq_len, num_nodes, in_channels)
        → [Permute to (batch, C, T, N)]
        → STConvBlock_1
        → STConvBlock_2
        → [Global avg pool over time]
        → FC output → (batch, num_nodes)

Reference:
    Yu et al. (2018). "Spatio-Temporal Graph Convolutional Networks:
    A Deep Learning Framework for Traffic Forecasting."
    https://arxiv.org/abs/1709.04875
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stgcn.layers import STConvBlock

logger = logging.getLogger(__name__)


class STGCN(nn.Module):
    """
    Full STGCN Model.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes (approach roads).
    in_channels : int
        Input features per node (e.g. 1 for vehicle count only).
    hidden_channels : int
        Hidden channel dimension in ST blocks.
    out_channels : int
        Output channel dimension of ST blocks.
    Ks : int
        Chebyshev polynomial order.
    cheb_polys : list of np.ndarray
        Pre-computed Chebyshev matrices.
    num_blocks : int
        Number of STConvBlocks to stack.
    kernel_size : int
        Temporal conv kernel size.
    dropout : float
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        Ks: int,
        cheb_polys: List[np.ndarray],
        num_blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels

        # Build stacked ST blocks
        blocks = []
        current_in = in_channels
        for i in range(num_blocks):
            block = STConvBlock(
                in_channels=current_in,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                Ks=Ks,
                cheb_polys=cheb_polys,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            blocks.append(block)
            current_in = out_channels  # next block takes previous out_channels

        self.st_blocks = nn.ModuleList(blocks)

        # Output layer: FC over combined (out_channels, num_nodes)
        self.output_fc = nn.Sequential(
            nn.Linear(out_channels * num_nodes, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_nodes),  # predict one value per node
        )

        self._init_weights()
        logger.info(
            f"STGCN initialized: nodes={num_nodes}, in={in_channels}, "
            f"hidden={hidden_channels}, out={out_channels}, "
            f"Ks={Ks}, blocks={num_blocks}"
        )

    def _init_weights(self):
        """Initialize FC layers with He initialization."""
        for m in self.output_fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, num_nodes, in_channels)
            Input traffic sequence.

        Returns
        -------
        out : torch.Tensor, shape (batch, num_nodes)
            Predicted traffic per node for next timestep.
        """
        # Permute to (batch, in_channels, seq_len, num_nodes)
        # Conv2d expects (batch, C, H, W) where H=time, W=nodes
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch, C, T, N)

        # Pass through ST blocks
        for block in self.st_blocks:
            x = block(x)  # (batch, out_channels, T, N)

        # Global average pooling over time dimension
        x = x.mean(dim=2)  # (batch, out_channels, N)

        # Flatten and project to output
        batch = x.shape[0]
        x = x.view(batch, -1)   # (batch, out_channels * N)
        out = self.output_fc(x)  # (batch, num_nodes)

        return out

    def predict(self, x: torch.Tensor, device: torch.device) -> np.ndarray:
        """
        Convenience inference method.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray — shape (1, seq_len, N, C)
        device : torch.device

        Returns
        -------
        prediction : np.ndarray, shape (num_nodes,)
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        x = x.to(device)
        with torch.no_grad():
            out = self.forward(x)
        return out.cpu().numpy().squeeze()


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def build_stgcn(config: dict, cheb_polys: List[np.ndarray]) -> STGCN:
    """
    Build STGCN from config dict.

    Parameters
    ----------
    config : dict — project config (from config.yaml)
    cheb_polys : list of (N, N) Chebyshev matrices from graph_builder

    Returns
    -------
    model : STGCN
    """
    stgcn_cfg = config.get("stgcn", {})
    num_nodes = config.get("intersection", {}).get("num_roads", 4)

    model = STGCN(
        num_nodes=num_nodes,
        in_channels=stgcn_cfg.get("in_channels", 1),
        hidden_channels=stgcn_cfg.get("hidden_channels", 16),
        out_channels=stgcn_cfg.get("out_channels", 64),
        Ks=stgcn_cfg.get("Ks", 3),
        cheb_polys=cheb_polys,
        num_blocks=stgcn_cfg.get("num_blocks", 2),
    )
    return model


if __name__ == "__main__":
    import sys, yaml
    sys.path.insert(0, ".")
    from preprocessing.graph_builder import build_graph
    import logging

    logging.basicConfig(level=logging.INFO)
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    graph = build_graph(cfg)
    model = build_stgcn(cfg, graph["cheb_polys"])

    print(f"\nModel architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Test forward pass
    batch, seq_len, N, C = 8, 12, 4, 1
    dummy = torch.randn(batch, seq_len, N, C)
    out = model(dummy)
    print(f"\nForward pass: {dummy.shape} → {out.shape}")
    assert out.shape == (batch, N), f"Unexpected output shape: {out.shape}"
    print("STGCN forward pass test passed ✓")
