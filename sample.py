import torch
from torch import Tensor


def sample_stratified(rays: Tensor, near=2., far=6., num_samples=64) -> tuple[Tensor, Tensor]:
    """Randomly samples points along the given rays by dividing the rays into even strata and randomly sampling one point per stratum.

    Args:
        rays (Tensor): A tensor of rays of shape (image_height, image_width, 3, 2), where the last dimension partitions the origins and directions.
        near (float, optional): The near plane.
            Defaults to 2
        far (float, optional): The far plane.
            Defaults to 6
        num_samples (int, optional): The number of samples (and the number of strata, as we randomly sample 1 point per stratum).
            Defaults to 64.

    Returns:
        Tensor: A tensor of points of shape (image_height, image_width, num_samples, 3).
    """
    # Each have shape (image_height, image_width, 3)
    rays_o, rays_d = rays[..., 0], rays[..., 1]

    # Calculate width of each stratum
    stratum_width = (far - near) / num_samples

    # Linearly interpolate between near and far plane
    # (image_height, image_width, num_samples)
    strata = torch.linspace(near, far - stratum_width,
                            num_samples).to(rays.device)

    # Randomly sample one point per stratum
    # [num_samples]
    t = strata + torch.rand_like(strata) * stratum_width

    # (image_height, image_width, num_samples, 3), [num_samples]
    return rays_o[..., None, :] + t[..., None] * rays_d[..., None, :], t


def sample_hierarchical(rays: Tensor, t: Tensor, weights: Tensor, num_samples=64) -> tuple[Tensor, Tensor]:
    """Samples points according to the pdf given by the weights.

    Args:
        rays (Tensor): A tensor of rays of shape (image_height, image_width, 3, 2), where the last dimension partitions the origins and directions.
        t (Tensor): A tensor of t values of shape (num_samples_stratified).
        weights (Tensor): A tensor of weights of shape (image_height, image_width, num_samples_stratified).
        num_samples (int, optional): The number of samples to take.
            Defaults to 64.

    Returns:
        Tensor: A tensor of t-values of shape (image_height, image_width, num_samples, 3).
    """
    rays_o, rays_d = rays[..., 0], rays[..., 1]

    # Generate pdf by normalizing weights
    # (image_height, image_width, num_samples_stratified)
    pdf = (weights / torch.sum(weights, dim=-1, keepdim=True)).to(t.device)

    # Generate cdf by cumulatively summing pdf
    # (image_height, image_width, num_samples_stratified)
    cdf = torch.cumsum(pdf, dim=-1)

    # Generate random samples from uniform distribution
    # (image_height, image_width, num_samples)
    u = torch.rand(cdf.shape[:-1] + (num_samples,)).to(t.device)

    # Find indices in the cdf of where u would be inserted
    # (image_height, image_width, num_samples)
    indices = torch.searchsorted(cdf, u)
    # Clamp indices to be within bounds
    indices = torch.clamp(indices, 0, cdf.shape[-1] - 1)

    # Find the t-values corresponding to the indices
    # (image_height, image_width, num_samples)
    # broadcast t to (image_height, image_width, num_samples_stratified)
    t = t[None, None, ...].expand_as(cdf)
    t_samples = torch.gather(t, dim=-1, index=indices)

    # Distance between consecutive t-values (width of each stratum)
    # [image_height, image_width, num_samples - 1]
    deltas = t[..., 1:] - t[..., :-1]
    # Add infinite distance (distance from last t-value to infinity)
    # [image_height, image_width, num_samples]
    deltas = torch.cat(
        [deltas, 1e10 * torch.ones_like(deltas[..., :1])], dim=-1)

    # Perturb t-values by a random amount within the stratum
    # (image_height, image_width, num_samples)
    t_samples += torch.rand_like(t_samples) * deltas

    # (image_height, image_width, num_samples, 3), (image_height, image_width, num_samples)
    return rays_o[..., None, :] + t_samples[..., None] * rays_d[..., None, :], t_samples
