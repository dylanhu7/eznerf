import html

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from jinja2 import Template

from dataloader.data import NeRFDataset, TestFrame
from model.model import NeRF
from rays.rays import get_rays
from sample.sample import sample_hierarchical, sample_stratified
from train.train import run_nerf


def test(model: NeRF,
         device: torch.device,
         test_loader: DataLoader[TestFrame()],
         output_dir: str) -> None:
    model.eval()
    dataset: NeRFDataset = test_loader.dataset  # type: ignore
    with torch.no_grad():
        loss_sum = 0.
        psnr_sum = 0.
        for batch_idx, frame in enumerate(test_loader):
            frame: TestFrame

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
            t, _ = torch.sort(t)
            image, _, _ = run_nerf(model, points, rays_d, t)

            image = image.permute(2, 0, 1)

            # (4, image_height, image_width)
            target_image = frame['image'].to(device)
            # Remove alpha channel: (4, H, W) to (3, H, W)
            target_image = target_image[:3, :, :]
            # Divide by 255.0 to normalize to [0, 1]
            target_image = target_image / 255.0

            loss = F.mse_loss(image, target_image)
            loss_sum += loss.item()
            psnr = 10 * torch.log10(1 / loss)
            psnr_sum += psnr.item()

            if batch_idx % 10 == 0:
                with open(file := f'{output_dir}/{epoch}_{epoch}.png', 'w+'):
                    image = image.detach()
                    image = image * 255.0 + 0.5
                    image = image.to(torch.uint8).cpu()
                    io.write_png(image, file)
                with open(file := f'{output_dir}/{batch_idx}_target.png', 'w+'):
                    target_image = target_image.detach()
                    target_image = target_image * 255.0 + 0.5
                    target_image = target_image.to(torch.uint8).cpu()
                    io.write_png(target_image, file)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx,
                    len(dataset),
                    100. * batch_idx / len(test_loader),
                    loss.item()))

        print(f'Average loss: {loss_sum / len(test_loader)}')
        print(f'Average PSNR: {psnr_sum / len(test_loader)}')


def generate_html(epoch: int, output_dir: str):
    with open('template.html.jinja', 'r') as f:
        template = Template(f.read())
    with open(f'{output_dir}/index.html', 'w+') as f:
        f.write(template.render(epoch=epoch, output_dir=output_dir))
