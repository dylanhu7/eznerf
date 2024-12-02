import torch
from torch import Tensor
import torch.nn.functional as F
from model.model import NeRF
from sample.sample import sample_hierarchical, sample_stratified


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


def query_nerf(
    model: NeRF, points: Tensor, ray_directions_normalized: Tensor
) -> tuple[Tensor, Tensor]:
    # [batch_size, 3] -> [batch_size, num_samples_stratified, 3]
    ray_directions_normalized = ray_directions_normalized[..., None, :].expand_as(
        points
    )
    return model(points, ray_directions_normalized)


def render_rays(
    coarse_model: NeRF,
    fine_model: NeRF,
    ray_origins: Tensor,
    ray_directions: Tensor,
    ray_directions_normalized: Tensor,
    train: bool,
) -> tuple[Tensor, Tensor]:
    # Sample points
    points, z = sample_stratified(ray_origins, ray_directions, 2, 6, 64)

    # Encode and forward pass
    rgb, sigma = query_nerf(coarse_model, points, ray_directions_normalized)

    # Volume render to obtain weights
    rgb_coarse, weights = volume_render(rgb, sigma, z)

    # Sample according to weights
    points_hierarchical, z_hierarchical = sample_hierarchical(
        ray_origins, ray_directions, z, weights, 128, train
    )
    z_hierarchical = z_hierarchical.detach()

    # Sort z values and gather points accordingly
    z, indices = torch.sort(torch.cat([z, z_hierarchical], -1), -1)
    points = torch.cat([points, points_hierarchical], -2)
    indices = indices[..., None].expand_as(points)
    points = torch.gather(points, -2, indices)

    # Encode and forward pass again
    rgb, sigma = query_nerf(fine_model, points, ray_directions_normalized)

    # Volume render to obtain final pixel colors
    rgb_fine, _ = volume_render(rgb, sigma, z)

    return rgb_coarse, rgb_fine
