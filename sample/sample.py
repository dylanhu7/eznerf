import torch
from torch import Tensor


def sample_stratified(
    ray_origins: Tensor,
    ray_directions: Tensor,
    near: float,
    far: float,
    num_samples: int,
) -> tuple[Tensor, Tensor]:
    """Randomly samples points along the given rays by dividing the rays into even strata and randomly sampling one point per stratum.

    Args:
        ray_origins (Tensor): A tensor of ray origins of shape [image_height, image_width, 3].
        ray_directions (Tensor): A tensor of non-normalized ray directions of shape [image_height, image_width, 3].
        near (float): The near plane.
        far (float): The far plane.
        num_samples (int): The number of samples (and the number of strata, as we randomly sample 1 point per stratum).

    Returns:
        Tensor: A tensor of points of shape [image_height, image_width, num_samples, 3].
        Tensor: A tensor of z values of shape [image_height, image_width, num_samples].
    """
    # Calculate width of each stratum
    stratum_width = (far - near) / num_samples

    # Linearly interpolate between near and far plane to get start of each stratum
    # [num_samples]
    strata_starts = torch.linspace(near, far - stratum_width, num_samples).to(
        ray_origins.device
    )

    # Expand strata_starts to match rays_o
    # [image_height, image_width, num_samples]
    strata_starts = strata_starts.expand(ray_origins.shape[:-1] + (num_samples,))

    # Randomly sample one point per stratum
    # [image_height, image_width, num_samples]
    z = strata_starts + torch.rand_like(strata_starts) * stratum_width

    # [image_height, image_width, num_samples, 3], [image_height, image_width, num_samples]
    return ray_origins[..., None, :] + z[..., :, None] * ray_directions[..., None, :], z


def sample_hierarchical(
    ray_origins: Tensor,
    ray_directions: Tensor,
    z: Tensor,
    weights: Tensor,
    num_samples: int,
    train: bool,
) -> tuple[Tensor, Tensor]:
    """Samples points according to the pdf given by the weights.

    Args:
        ray_origins (Tensor): A tensor of ray origins of shape [image_height, image_width, 3].
        ray_directions (Tensor): A tensor of non-normalized ray directions of shape [image_height, image_width, 3].
        z (Tensor): A tensor of z values of shape [image_height, image_width, num_samples_stratified].
        weights (Tensor): A tensor of weights of shape [image_height, image_width, num_samples_stratified - 1].
        num_samples (int): The number of samples to take.

    Returns:
        Tensor: A tensor of t-values of shape [image_height, image_width, num_samples, 3].
    """
    # [image_height, image_width, num_samples_stratified - 1]
    deltas = z[..., 1:] - z[..., :-1]

    # Generate pdf by normalizing weights
    # [image_height, image_width, num_samples_stratified - 1]
    pmf = weights / torch.sum(weights, dim=-1, keepdim=True)

    # Generate cdf by cumulatively summing pdf
    # [image_height, image_width, num_samples_stratified - 1]
    cdf = torch.cumsum(pmf, dim=-1)

    # Generate random samples from uniform distribution
    # [image_height, image_width, num_samples]
    if train:
        u = torch.rand(cdf.shape[:-1] + (num_samples,))
    else:
        u = torch.linspace(0.0, 1.0, steps=num_samples)
        u = u.expand(cdf.shape[:-1] + (num_samples,))

    # Find indices in the cdf of where u would be inserted
    # [image_height, image_width, num_samples]
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u)
    # Clamp indices to be within bounds
    indices = torch.clamp(indices, 0, cdf.shape[-1] - 1)

    # Find the t-values corresponding to the indices
    # [image_height, image_width, num_samples]
    z = torch.gather(z, dim=-1, index=indices)
    # [image_height, image_width, num_samples]
    deltas = torch.gather(deltas, dim=-1, index=indices)

    # Perturb t values by a random amount within the stratum
    # [image_height, image_width, num_samples]
    # if train:
    # z += torch.rand_like(z) * deltas

    # [image_height, image_width, num_samples, 3], [image_height, image_width, num_samples]
    return ray_origins[..., None, :] + z[..., None] * ray_directions[..., None, :], z
