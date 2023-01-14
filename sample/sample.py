import torch
from torch import Tensor


def sample_stratified(rays: Tensor, near: float, far: float, num_samples: int) -> tuple[Tensor, Tensor]:
    """Randomly samples points along the given rays by dividing the rays into even strata and randomly sampling one point per stratum.

    Args:
        rays (Tensor): A tensor of rays of shape [image_height, image_width, 3, 2], where the last dimension partitions the origins and directions.
        near (float): The near plane.
        far (float): The far plane.
        num_samples (int): The number of samples (and the number of strata, as we randomly sample 1 point per stratum).

    Returns:
        Tensor: A tensor of points of shape [image_height, image_width, num_samples, 3].
    """
    # Each have shape [image_height, image_width, 3]
    rays_o, rays_d = rays[..., 0], rays[..., 1]

    # Calculate width of each stratum
    stratum_width = (far - near) / num_samples

    # Linearly interpolate between near and far plane
    # [num_samples]
    strata = torch.linspace(near, far - stratum_width,
                            num_samples).to(rays.device)

    # Randomly sample one point per stratum
    # [num_samples]
    t = strata + torch.rand_like(strata) * stratum_width

    # [image_height, image_width, num_samples, 3], [num_samples]
    return rays_o[..., None, :] + t[..., None] * rays_d[..., None, :], t


def sample_hierarchical(rays: Tensor, t: Tensor, deltas: Tensor, weights: Tensor, num_samples: int) -> tuple[Tensor, Tensor]:
    """Samples points according to the pdf given by the weights.

    Args:
        rays (Tensor): A tensor of rays of shape [image_height, image_width, 3, 2], where the last dimension partitions the origins and directions.
        t (Tensor): A tensor of t values of shape [num_samples_stratified].
        delta: A tensor of delta values of shape [num_samples_stratified - 1].
        weights (Tensor): A tensor of weights of shape [image_height, image_width, num_samples_stratified - 1].
        num_samples (int): The number of samples to take.

    Returns:
        Tensor: A tensor of t-values of shape [image_height, image_width, num_samples, 3].
    """
    rays_o, rays_d = rays[..., 0], rays[..., 1]
    # [image_height, image_width, num_samples_stratified]
    t = t[None, None, :-1].expand_as(weights)
    # [image_height, image_width, num_samples_stratified - 1]
    deltas = deltas[None, None, ...].expand_as(weights)

    # Generate pdf by normalizing weights
    # [image_height, image_width, num_samples_stratified - 1]
    pmf = (weights / torch.sum(weights, dim=-1, keepdim=True)).to(t.device)

    # Generate cdf by cumulatively summing pdf
    # [image_height, image_width, num_samples_stratified - 1]
    cdf = torch.cumsum(pmf, dim=-1)

    # Generate random samples from uniform distribution
    # [image_height, image_width, num_samples]
    u = torch.rand(cdf.shape[:-1] + (num_samples,)).to(t.device)

    # Find indices in the cdf of where u would be inserted
    # [image_height, image_width, num_samples]
    indices = torch.searchsorted(cdf, u)
    # Clamp indices to be within bounds
    indices = torch.clamp(indices, 0, cdf.shape[-1] - 1)

    # Find the t-values corresponding to the indices
    # [image_height, image_width, num_samples]
    t = torch.gather(t, dim=-1, index=indices)
    # [image_height, image_width, num_samples]
    deltas = torch.gather(deltas, dim=-1, index=indices)

    # Perturb t values by a random amount within the stratum
    # [image_height, image_width, num_samples]
    t += torch.rand_like(t) * deltas

    # [image_height, image_width, num_samples, 3], [image_height, image_width, num_samples]
    return rays_o[..., None, :] + t[..., None] * rays_d[..., None, :], t
