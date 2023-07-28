import torch
import torch.nn as nn
from torch import Tensor

from encoder.encoder import Encoder


class NeRF(nn.Module):
    def __init__(
        self,
        x_num_bands: int = 10,
        d_num_bands: int = 4,
        hidden_dim: int = 256,
    ):
        super(NeRF, self).__init__()
        self.x_num_bands = x_num_bands
        self.d_num_bands = d_num_bands
        self.hidden_dim = hidden_dim
        # 3-dimensional raw xyz coordinates and x_num_bands * 2 (sin and cos) encoded coordinates
        self.x_dim = 3 + 3 * 2 * x_num_bands
        self.d_dim = 3 + 3 * 2 * d_num_bands

        def relu_layer(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())

        self.x_encoder = Encoder(self.x_num_bands)
        self.d_encoder = Encoder(self.d_num_bands)

        self.relu_block_1 = nn.Sequential(
            relu_layer(self.x_dim, hidden_dim),
            relu_layer(hidden_dim, hidden_dim),
            relu_layer(hidden_dim, hidden_dim),
            relu_layer(hidden_dim, hidden_dim),
            relu_layer(hidden_dim, hidden_dim),
        )

        self.relu_block_2 = nn.Sequential(
            relu_layer(hidden_dim + self.x_dim, hidden_dim),
            relu_layer(hidden_dim, hidden_dim),
            relu_layer(hidden_dim, hidden_dim),
        )

        self.sigma_out = nn.Linear(hidden_dim, 1)

        self.bottleneck = nn.Linear(hidden_dim, hidden_dim)

        self.rgb_out = nn.Sequential(
            relu_layer(hidden_dim + self.d_dim, hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor, d: Tensor) -> tuple[Tensor, Tensor]:
        # Positional encoding
        x = self.x_encoder(x)
        d = self.d_encoder(d)

        # MLP
        output = x
        output = self.relu_block_1(output)
        output = torch.cat([x, output], dim=-1)
        output = self.relu_block_2(output)
        sigma = self.sigma_out(output).squeeze(-1)
        output = self.bottleneck(output)
        output = torch.cat([output, d], dim=-1)
        rgb = self.rgb_out(output)
        return rgb, sigma
