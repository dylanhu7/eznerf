import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from encoder.encoder import Encoder


class NeRF(nn.Module):
    def __init__(
        self,
        x_num_bands: int,
        d_num_bands: int,
        hidden_dim: int = 256,
        make_encoders: bool = True
    ):
        super(NeRF, self).__init__()
        self.x_num_bands = x_num_bands
        self.d_num_bands = d_num_bands
        self.hidden_dim = hidden_dim
        # 3-dimensional raw xyz coordinates and x_num_bands * 2 (sin and cos) encoded coordinates
        self.x_dim = 3 + 3 * 2 * x_num_bands
        self.d_dim = 3 + 3 * 2 * d_num_bands

        if make_encoders:
            self.x_encoder = Encoder(x_num_bands)
            self.d_encoder = Encoder(d_num_bands)

        def relu_layer(in_features: int, out_features: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU()
            )

        self.layers = nn.ModuleDict(
            {
                "layer_1": relu_layer(self.x_dim, hidden_dim),
                "layer_2": relu_layer(hidden_dim, hidden_dim),
                "layer_3": relu_layer(hidden_dim, hidden_dim),
                "layer_4_skipx": relu_layer(hidden_dim, hidden_dim),
                "layer_5": relu_layer(hidden_dim + self.x_dim, hidden_dim),
                "layer_6": relu_layer(hidden_dim, hidden_dim),
                "layer_7": relu_layer(hidden_dim, hidden_dim),
                "layer_8_skipd_sigmaout": nn.Linear(hidden_dim, hidden_dim + 1),
                "layer_9": relu_layer(hidden_dim + self.d_dim, hidden_dim // 2),
                "layer_11": nn.Sequential(
                    nn.Linear(hidden_dim // 2, 3),
                    nn.Sigmoid(),
                )
            }
        )

    def forward(self, input: Tensor) -> tuple[Tensor, Tensor]:
        x, d = torch.split(input, [self.x_dim, self.d_dim], dim=-1)
        output = x
        sigma = torch.empty(0)
        for layer_name in self.layers:
            layer = self.layers[layer_name]
            output: Tensor = layer(output)
            if "sigmaout" in layer_name:
                output, sigma = output[..., :-1], F.relu(output[..., -1:])
            if "skipx" in layer_name:
                output = torch.cat([output, x], dim=-1)
            if "skipd" in layer_name:
                output = torch.cat([output, d], dim=-1)
        return output, sigma
