import os
from typing import Callable, TypedDict
from jinja2 import Template

import torch
import torch.backends.mps
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import io

from dataloader.data import Frame, NeRFDataset, TestFrame
from model.model import NeRF
from rays.rays import get_rays
from render.render import volume_render
from sample.sample import sample_hierarchical, sample_stratified


class ResultDict(TypedDict):
    batch: int
    loss: float
    psnr: float
    image: str
    target_image: str


def run_func(model: NeRF,
             device: torch.device,
             loader: DataLoader[Frame] | DataLoader[TestFrame],
             optimizer: Optimizer,
             output_dir: str,
             train: bool) -> Callable[[int], None]:
    losses: list[float] = []
    psnrs: list[float] = []

    def run_epoch(epoch: int):
        loss_sum = 0.
        psnr_sum = 0.
        # make epoch dir if it doesn't exist
        epoch_dir = os.path.join(output_dir, f'epoch_{epoch}')
        images_dir = os.path.join(epoch_dir, 'images')
        if not train and not os.path.exists(images_dir):
            os.makedirs(images_dir)
        results: list[ResultDict] = []
        dataset: NeRFDataset[Frame] = loader.dataset  # type: ignore

        model.train(train)
        with torch.set_grad_enabled(train):
            for batch_idx, frame in enumerate(loader):
                frame: Frame

                # (image_height, image_width, 3, 2)
                rays = get_rays(dataset.image_width,
                                dataset.image_height,
                                dataset.camera_angle_x,
                                frame['transform_matrix'].to(device))
                # (image_height, image_width, 3)
                rays_d = rays[..., 1]

                # (H, W, num_samples, 3), (num_samples)
                points, t = sample_stratified(rays, 2., 6., 64)
                _, weights, deltas = run_nerf(model, points, rays_d, t)

                # (H, W, num_samples, 3), (H, W, num_samples)
                points_hierarchical, t_hierarchical = sample_hierarchical(
                    rays, t, deltas, weights, 128)

                points = torch.cat([points, points_hierarchical], dim=2)
                t = t[None, None, :].expand(
                    t_hierarchical.shape[:-1] + (t.shape[-1],))
                t = torch.cat([t, t_hierarchical], dim=-1)
                t, indices = torch.sort(t)
                points = torch.gather(points, dim=-2, index=indices[..., None].expand(
                    indices.shape + (points.shape[-1],)))
                image, _, _ = run_nerf(model, points, rays_d, t)

                image = image.permute(2, 0, 1)

                # (4, image_height, image_width)
                target_image = frame['image'].to(device)
                # Remove alpha channel: (4, H, W) to (3, H, W)
                target_image = target_image[:3, :, :]
                # Divide by 255.0 to normalize to [0, 1]
                target_image = target_image / 255.0

                loss = F.mse_loss(image, target_image)

                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    losses.append(loss_item := loss.item())
                    psnrs.append(
                        psnr := 10. * torch.log10(Tensor(1. / loss)).item())
                    loss_sum += loss.item()
                    psnr_sum += psnr

                    if batch_idx % 10 == 0:
                        image_file = os.path.join("images", f'{batch_idx}.png')
                        target_image_file = os.path.join(
                            "images", f'{batch_idx}_target.png')
                        image_path = os.path.join(epoch_dir, image_file)
                        target_image_path = os.path.join(epoch_dir, target_image_file)
                        with open(image_path, 'w+') as file:
                            image = image.detach()
                            image = image * 255.0 + 0.5
                            image = image.to(torch.uint8).cpu()
                            io.write_png(image, file.name)
                        with open(target_image_path, 'w+') as file:
                            target_image = target_image.detach()
                            target_image = target_image * 255.0 + 0.5
                            target_image = target_image.to(torch.uint8).cpu()
                            io.write_png(target_image, file.name)
                        results.append(ResultDict(
                            batch=batch_idx,
                            loss=loss_item,
                            psnr=psnr,
                            image=image_file,
                            target_image=target_image_file))

        if not train:
            generate_html(epoch, epoch_dir, results)

    return run_epoch


def run_nerf(model: NeRF,
             points: Tensor,
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


def generate_html(epoch: int, epoch_dir: str, results: list[ResultDict]):
    with open(os.path.join(os.path.dirname(__file__), 'template.html.jinja')) as f:
        template = Template(f.read())
    with open(f'{epoch_dir}/index.html', 'w+') as f:
        f.write(template.render(epoch=epoch, results=results))
