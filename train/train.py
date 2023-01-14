from typing import Callable

import torch
import torch.backends.mps
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import io

from dataloader.data import Frame, NeRFDataset
from model.model import NeRF
from rays.rays import get_rays
from render.render import volume_render
from sample.sample import sample_hierarchical, sample_stratified


def train_func(model: NeRF,
               device: torch.device,
               train_loader: DataLoader[Frame],
               optimizer: Optimizer) -> Callable[[int], None]:
    def run_nerf(points: Tensor,
                 rays_d: Tensor,
                 t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x_encoded = model.x_encoder(points)
        d_encoded = model.d_encoder(rays_d)
        # (H, W, num_samples, 3 + 3 * 2 * d_num_bands)
        d_encoded: torch.Tensor = d_encoded[..., None, :].expand(
            x_encoded.shape[:-1] + (d_encoded.shape[-1],))

        input = torch.cat([x_encoded, d_encoded], dim=-1)
        # (H * W * num_samples, 3 + 3 * 2 * (x_num_bands + d_num_bands))
        input = input.reshape(-1, input.shape[-1])

        output = model(input)

        # (H * W * num_samples, 3), (H * W * num_samples)
        rgb, sigma = output[..., :3], output[..., 3]
        # (H, W, num_samples, 3)
        rgb = rgb.reshape(
            (list(points.shape)[:-1] + [3]))
        # (H, W, num_samples)
        sigma = sigma.reshape(
            (list(points.shape)[:-1]))

        # (H, W, num_samples_stratified, 3), (H, W, num_samples_stratified)
        return volume_render(rgb, sigma, t)

    def train_epoch(epoch: int):
        model.train()
        dataset: NeRFDataset = train_loader.dataset  # type: ignore
        for batch_idx, frame in enumerate(train_loader):
            frame: Frame

            # (image_height, image_width, 3, 2)
            rays = get_rays(dataset.image_width, dataset.image_height,
                            dataset.camera_angle_x, frame['transform_matrix'].to(device))
            # (image_height, image_width, 3)
            rays_d = rays[..., 1]

            # (H, W, num_samples, 3), (num_samples)
            points, t = sample_stratified(rays, 2., 6., 64)
            _, weights, deltas = run_nerf(points, rays_d, t)

            # (H, W, num_samples, 3), (H, W, num_samples)
            points_hierarchical, t_hierarchical = sample_hierarchical(
                rays, t, deltas, weights, 128)

            points = torch.cat([points, points_hierarchical], dim=2)
            t = t[None, None, :].expand(
                t_hierarchical.shape[:-1] + (t.shape[-1],))
            t = torch.cat([t, t_hierarchical], dim=-1)
            t, _ = torch.sort(t)
            # t, _ = torch.sort(t_hierarchical)
            # points = points_hierarchical
            image, _, _ = run_nerf(points, rays_d, t)

            image = image.permute(2, 0, 1)

            # (4, image_height, image_width)
            target_image = frame['image'].to(device)
            # Remove alpha channel: (4, H, W) to (3, H, W)
            target_image = target_image[:3, :, :]
            # Divide by 255.0 to normalize to [0, 1]
            target_image = target_image / 255.0

            loss = F.mse_loss(image, target_image)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                with open(file := f'output/{batch_idx}.png', 'w+'):
                    image = image.detach()
                    image = image * 255.0 + 0.5
                    image = image.to(torch.uint8).cpu()
                    io.write_png(image, file)
                with open(file := f'output/{batch_idx}_target.png', 'w+'):
                    target_image = target_image.detach()
                    target_image = target_image * 255.0 + 0.5
                    target_image = target_image.to(torch.uint8).cpu()
                    io.write_png(target_image, file)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    len(dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))
    return train_epoch
