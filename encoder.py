import torch


class Encoder(torch.nn.Module):
    def __init__(self, num_bands: int, max_freq: float):
        super().__init__()
        self.num_bands = num_bands
        self.max_freq = max_freq
        self.bands = torch.linspace(2. ** 0., 2. ** max_freq, num_bands)
        self.encoding_fns = [lambda x: x]
        for band in self.bands:
            self.encoding_fns.append(lambda x, band=band: torch.sin(x * band))
            self.encoding_fns.append(lambda x, band=band: torch.cos(x * band))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(x) for fn in self.encoding_fns], dim=-1)
