import torch
from torch import Tensor


def volume_render(color: Tensor,
                  sigma: Tensor,
                  t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
    # [image_height, image_width, num_samples - 1]
    deltas = t[..., 1:] - t[..., :-1]

    # Omit values for last sample, as we don't have a delta for it.
    # [image_height, image_width, num_samples - 1, 3]
    color = color[..., :-1, :]
    # [image_height, image_width, num_samples - 1]
    sigma = sigma[..., :-1]

    # Calculate alpha values (alpha_i from the paper).
    # [image_height, image_width, num_samples - 1]
    alphas = 1. - torch.exp(-sigma * deltas)

    # Calculate the transmittance (T_i from the paper).
    # [image_height, image_width, num_samples - 1]
    transmittance = cumprod_exclusive(1. - alphas)

    # Calculate weights for each sample as product of transmittance and alpha.
    # (w_i from equation 5 in the paper)
    # [image_height, image_width, num_samples - 1]
    weights = transmittance * alphas

    image = torch.sum(weights[..., None] * color, dim=-2)

    return image, weights, deltas


def cumprod_exclusive(tensor: Tensor) -> Tensor:
    """An exclusive cumulative product"""
    # [image_height, image_width, num_samples]
    return torch.cumprod(torch.cat(
        [torch.ones_like(tensor[..., :1]), tensor[..., :-1]], dim=-1), dim=-1)
