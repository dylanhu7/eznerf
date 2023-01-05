import torch
from torch import Tensor


def volume_render(rgb: Tensor, sigma: Tensor, t_vals: Tensor, white_bkgd=False) -> tuple[Tensor, Tensor]:
    """Volume renders a radiance field.

    Args:
        rgb (Tensor): [image_height, image_width, num_samples, 3] The RGB color of each point.
        sigma (Tensor): [image_height, image_width, num_samples] The Volume density of each point.
        t_vals (Tensor): [num_samples] The t-values along each ray.
        white_bkgd (bool, optional): Whether to render a white background.
            Defaults to False (black background).
    Returns:
        Tensor: [image_height, image_width, 3] RGB color of each pixel.
    """
    # Distance between consecutive z-values (delta_i from equation 3 in the paper).
    # [num_samples - 1]
    deltas = t_vals[..., 1:] - t_vals[..., :-1]
    # Add infinite distance (distance from last z-value to infinity)
    # [num_samples]
    deltas = torch.cat(
        [deltas, 1e10 * torch.ones_like(deltas[..., :1])], dim=-1)

    # Calculate alpha values (alpha_i from the paper).
    # [image_height, image_width, num_samples]
    alphas = 1.0 - torch.exp(-sigma * deltas)

    def cumprod_exclusive(tensor: Tensor) -> Tensor:
        """Cumulative product excluding the first element."""
        # [image_height, image_width, num_samples]
        return torch.cumprod(torch.cat(
            [torch.ones_like(tensor[..., :1]), tensor[..., :-1]], dim=-1), dim=-1)

    # Calculate the transmittance (T_i from the paper).
    # [image_height, image_width, num_samples]
    transmittance = cumprod_exclusive(1.0 - alphas)

    # Calculate weights for each sample as product of transmittance and alpha.
    # (w_i from equation 5 in the paper)
    # [image_height, image_width, num_samples]
    weights = transmittance * alphas

    image = torch.sum(weights[..., None] * rgb, dim=-2)

    return image, weights
