import torch
from torch import Tensor
import torch.nn.functional as F


def volume_render(
    color: Tensor,
    sigma: Tensor,
    t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Volume renders a radiance field.

    Args:
        color (Tensor): [image_height, image_width, num_samples, 3]
            The RGB color of each point.
        sigma (Tensor): [image_height, image_width, num_samples]
            The volume density of each point.
        t (Tensor): [num_samples]
            The t-values along each ray.
    Returns:
        Tensor: [image_height, image_width, 3]
            The RGB color of each pixel.
    """
    # Distance between consecutive z-values (delta_i from equation 3 in the paper).
    # [batch_size, num_samples - 1]
    deltas = t[..., 1:] - t[..., :-1]

    # Omit values for last sample, as we don't have a delta for it.
    # [image_height, image_width, num_samples - 1, 3]
    color = color[..., :-1, :]
    # [batch_size, num_samples - 1]
    sigma = sigma[..., :-1]

    sigma = F.relu(sigma)

    # Calculate alpha values (alpha_i from the paper).
    # [image_height, image_width, num_samples - 1]
    alphas = 1.0 - torch.exp(-sigma * deltas)

    # Calculate the transmittance (T_i from the paper).
    # [image_height, image_width, num_samples - 1]
    transmittance = cumprod_exclusive(1.0 - alphas, dim=-1)

    # Calculate weights for each sample as product of transmittance and alpha.
    # (w_i from equation 5 in the paper)
    # [image_height, image_width, num_samples - 1]
    weights = transmittance * alphas

    image = torch.sum(weights[..., None] * color, dim=-2)
    return image, weights


def cumprod_exclusive(x: Tensor, dim: int) -> Tensor:
    """An exclusive cumulative product"""
    # [image_height, image_width, num_samples]
    return torch.cumprod(
        torch.cat([torch.ones_like(x[..., :1]), x[..., :-1]], dim=dim), dim=dim
    )
