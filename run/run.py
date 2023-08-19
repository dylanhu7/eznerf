import PIL.Image as Image
import torch
import torch.backends.mps
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from config.config import EZNeRFConfig
from model.model import NeRF
from render.render import volume_render
from sample.sample import (
    sample_hierarchical,
    sample_stratified,
)


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


def query_nerf(
    model: NeRF, points: Tensor, ray_directions_normalized: Tensor
) -> tuple[Tensor, Tensor]:
    # [batch_size, 3] -> [batch_size, num_samples_stratified, 3]
    ray_directions_normalized = ray_directions_normalized[..., None, :].expand_as(
        points
    )
    return model(points, ray_directions_normalized)


def train_batch(
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    def train(batch: tuple[Tensor, ...]):
        (
            ray_origins,
            ray_directions,
            ray_directions_normalized,
            target_image,
        ) = batch

        rgb_coarse, rgb_fine = render_rays(
            coarse_model,
            fine_model,
            ray_origins,
            ray_directions,
            ray_directions_normalized,
            train=True,
        )
        coarse_loss = F.mse_loss(rgb_coarse, target_image)
        fine_loss = F.mse_loss(rgb_fine, target_image)
        loss = coarse_loss + fine_loss
        psnr = -10 * torch.log10(fine_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return loss, psnr, coarse_loss, fine_loss

    return train


def test_func(config: EZNeRFConfig, coarse_model: NeRF, fine_model: NeRF):
    @torch.no_grad()
    def test(batch: tuple[Tensor, ...]):
        (
            ray_origins,
            ray_directions,
            ray_directions_normalized,
            target_image,
        ) = batch
        chunk_size = config.batch_size * 2
        chunk_count = ray_origins.shape[0] // chunk_size + 1
        chunk_rays_o = torch.chunk(ray_origins, chunk_count)
        chunk_rays_d = torch.chunk(ray_directions, chunk_count)
        chunk_rays_d_norm = torch.chunk(ray_directions_normalized, chunk_count)
        image = []
        for i in range(chunk_count):
            _, rgb_fine = render_rays(
                coarse_model,
                fine_model,
                chunk_rays_o[i],
                chunk_rays_d[i],
                chunk_rays_d_norm[i],
                train=False,
            )
            image.append(rgb_fine)

        image = torch.cat(image, dim=0)
        loss = F.mse_loss(image, target_image)
        psnr = -10 * torch.log10(loss)
        image = image.reshape(config.H, config.W, 3)
        image = (torch.clip(image, 0, 1) * 255).to(torch.uint8).cpu().numpy()
        image = Image.fromarray(image)
        return image, loss, psnr

    return test
