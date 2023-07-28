import os
from typing import Callable, TypedDict

import torch
import torch.backends.mps
import torch.nn.functional as F
from jinja2 import Template
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import io
from tqdm import tqdm

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


def train_func(
    coarse_model: NeRF, fine_model: NeRF, loader: DataLoader, optimizer: Optimizer
) -> Callable[[int], None]:
    def train_epoch(epoch: int):
        for batch in (pbar := tqdm(loader)):
            batch: tuple[Tensor, ...]
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
            pbar.set_description(
                f"[Training | Epoch {epoch} | Loss: {loss.item():.4f} | PSNR: {psnr.item():.4f}]"
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return train_epoch


class ResultDict(TypedDict):
    batch: int
    loss: float
    psnr: float
    image: str
    target_image: str


def test_func(
    coarse_model: NeRF, fine_model: NeRF, loader: DataLoader, output_dir: str
) -> Callable[[int], None]:
    def test_epoch(epoch: int):
        losses: list[float] = []
        psnrs: list[float] = []
        loss_sum = 0.0
        psnr_sum = 0.0
        epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
        images_dir = os.path.join(epoch_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        results: list[ResultDict] = []

        with torch.no_grad():
            for frame_idx, frame in enumerate(pbar := tqdm(loader)):
                if frame_idx % 10 == 0:
                    frame: tuple[Tensor, ...]
                    (
                        ray_origins,
                        ray_directions,
                        ray_directions_normalized,
                        target_image,
                    ) = frame
                    chunk_size = ray_origins.shape[0] // 20
                    chunk_count = ray_origins.shape[0] // chunk_size
                    chunk_rays_o = torch.chunk(ray_origins, chunk_count)
                    chunk_rays_d = torch.chunk(ray_directions, chunk_count)
                    chunk_rays_d_norm = torch.chunk(
                        ray_directions_normalized, chunk_count
                    )
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
                    losses.append(loss_item := loss.item())
                    psnrs.append(psnr := (-10 * torch.log10(loss)).item())
                    loss_sum += loss_item
                    psnr_sum += psnr
                    if frame_idx % 10 == 0:
                        results.append(
                            write_output(
                                image.reshape(400, 400, 3),
                                target_image.reshape(400, 400, 3),
                                frame_idx,
                                epoch_dir,
                                loss_item,
                                psnr,
                            )
                        )
                    pbar.set_description(
                        f"[Testing | Epoch {epoch} | Frame {frame_idx}/{len(loader)} | Loss: {loss_item:.4f} | PSNR: {psnr:.4f}]"
                    )
        generate_html(epoch, epoch_dir, results)

    return test_epoch


def write_output(
    image: Tensor,
    target_image: Tensor,
    batch_idx: int,
    epoch_dir: str,
    loss: float,
    psnr: float,
) -> ResultDict:
    image_file = os.path.join("images", f"{batch_idx}.png")
    target_image_file = os.path.join("images", f"{batch_idx}_target.png")
    image_path = os.path.join(epoch_dir, image_file)
    target_image_path = os.path.join(epoch_dir, target_image_file)
    with open(image_path, "w+") as file:
        image = image.permute(2, 0, 1)
        image = image * 255.0
        image = image.to(torch.uint8).cpu()
        io.write_png(image, file.name)
    with open(target_image_path, "w+") as file:
        target_image = target_image.permute(2, 0, 1)
        target_image = target_image * 255.0
        target_image = target_image.to(torch.uint8).cpu()
        io.write_png(target_image, file.name)
    return ResultDict(
        batch=batch_idx,
        loss=loss,
        psnr=psnr,
        image=image_file,
        target_image=target_image_file,
    )


def generate_html(epoch: int, epoch_dir: str, results: list[ResultDict]):
    with open(os.path.join(os.path.dirname(__file__), "template.html.jinja")) as f:
        template = Template(f.read())
    with open(f"{epoch_dir}/index.html", "w+") as f:
        f.write(template.render(epoch=epoch, results=results))
