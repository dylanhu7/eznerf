import os
import torch
import torch.utils
import torch.utils.data
import torch.backends.mps
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from torch import _dynamo
from torchvision import io

from model import NeRF
from data import get_data_loader, Frame, NeRFDataset
from encoder import Encoder
from rays import get_rays
from sample import sample_stratified
from render import volume_render


def train_func(model: NeRF, device: torch.device, train_loader: torch.utils.data.DataLoader[Frame], optimizer: torch.optim.Optimizer):
    def train_epoch(epoch: int):
        model.train()
        dataset: NeRFDataset = train_loader.dataset  # type: ignore
        for batch_idx, frame in enumerate(train_loader):
            if batch_idx == 1:
                break
            frame: Frame

            rays = get_rays(dataset.image_width, dataset.image_height,
                            dataset.camera_angle_x, frame['transform_matrix']).to(device)  # (image_height, image_width, 3, 2)
            rays_d = rays[..., 1].to(device)
            # normalize the directions
            rays_d = rays_d / \
                torch.norm(rays_d, dim=-1, keepdim=True).to(device)

            # (H, W, num_samples, 3), (H, W, num_samples)
            points, t = sample_stratified(rays)
            points = points.to(device)
            t = t.to(device)

            # (H, W, num_samples, 3 + 3 * 2 * x_num_bands)
            x_encoded = model.x_encoder(points).to(device)
            # (H, W, 3 + 3 * 2 * d_num_bands)
            d_encoded: torch.Tensor = model.d_encoder(rays_d).to(device)
            # (H, W, num_samples, 3 + 3 * 2 * d_num_bands)
            d_encoded: torch.Tensor = d_encoded[..., None, :].expand(
                x_encoded.shape[:-1] + (d_encoded.shape[-1],)).to(device)

            input = torch.cat([x_encoded, d_encoded], dim=-1).to(device)
            # (H * W * num_samples, 3 + 3 * 2 * (x_num_bands + d_num_bands)
            input = input.reshape(-1, input.shape[-1]).to(device)

            output = []
            batch_size = 2048
            for i in range(0, input.shape[0], batch_size):
                output.append(model(input[i:i+batch_size]).to(device))
            output = torch.cat(output, dim=0).to(device)

            # (H * W * num_samples, 3), (H * W * num_samples)
            rgb, sigma = output[..., :3].to(device), output[..., 3].to(device)
            # (H, W, num_samples, 3)
            rgb = rgb.reshape(
                (points.shape[0], points.shape[1], points.shape[2], 3)).to(device)
            # (H, W, num_samples)
            sigma = sigma.reshape(
                (points.shape[0], points.shape[1], points.shape[2])).to(device)

            # (H, W, 3)
            image = volume_render(rgb, sigma, t, rays_d).to(device)
            # (3, H, W)
            image = image.permute(2, 0, 1)

            # (4, image_height, image_width)
            target_image = frame['image'].to(device)
            # Remove alpha channel: (4, H, W) to (3, H, W)
            target_image = target_image[:3, :, :]
            # Divide by 255.0 to normalize to [0, 1]
            target_image = target_image / 255.0

            loss = F.mse_loss(image, target_image).to(device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                # make output directory if it doesn't exist
                if not os.path.exists('output'):
                    os.makedirs('output')
                with open(file := f'output/{batch_idx}.png', 'w+'):
                    image = image * 255.0 + 0.5
                    image = image.to(torch.uint8)
                    image = image.cpu()
                    io.write_png(image, file)
                with open(file := f'output/{batch_idx}_target.png', 'w+'):
                    target_image = target_image * 255.0
                    target_image = target_image.to(torch.uint8).cpu()
                    io.write_png(target_image, file)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(dataset), 100. * batch_idx / len(train_loader), loss.item()))
    return train_epoch


def main():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    # torch._dynamo.config.verbose = True
    # torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.suppress_errors = True
    model = NeRF(10, 4).to(device)
    train_loader = get_data_loader(
        "data/nerf_synthetic/lego/transforms_train_64x64.json", train=True, shuffle=False, batch_size=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train = train_func(model, device, train_loader, optimizer)

    # Train the model
    for epoch in range(1000):
        train(epoch)
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_{epoch}.pt')


if __name__ == '__main__':
    main()
