import torch


class Encoder(torch.nn.Module):
    def __init__(self, num_bands: int):
        super().__init__()
        self.num_bands = num_bands
        self.encoding_fns = [lambda x: x]
        for i in range(self.num_bands):
            self.encoding_fns.append(
                lambda x, i=i: torch.sin(2 ** i * torch.pi * x))
            self.encoding_fns.append(
                lambda x, i=i: torch.cos(2 ** i * torch.pi * x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(x) for fn in self.encoding_fns], dim=-1)
