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
    # (image_height, image_width, num_samples)
    t = strata + torch.rand_like(strata) * stratum_width

    # (image_height, image_width, num_samples, 3)
    return rays_o[..., None, :] + t[..., None] * rays_d[..., None, :], t
