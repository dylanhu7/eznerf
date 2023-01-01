import torch
import torch.utils
import torch.utils.data
import torch.backends.mps
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import NeRF
from data import get_data_loader, Frame, NeRFDataset
from encoder import Encoder
from rays import get_rays
from sample import sample_stratified
from render import volume_render


def train(model: NeRF, device: torch.device, train_loader: torch.utils.data.DataLoader[Frame], optimizer: torch.optim.Optimizer, epoch: int):
    model.train()
    dataset: NeRFDataset = train_loader.dataset  # type: ignore
    idx = 0
    for batch_idx, frame in enumerate(train_loader):
        print(f'batch_idx: {batch_idx}')
        # if idx == 1:
        # break
        frame: Frame
        target_image = frame['image'].to(device)
        rays = get_rays(dataset.image_width, dataset.image_height,
                        dataset.camera_angle_x, frame['transform_matrix']).to(device)  # (image_height, image_width, 3, 2)
        rays_o, rays_d = rays[..., 0], rays[..., 1].to(device)
        x_encoder = Encoder(model.x_num_bands).to(device)
        d_encoder = Encoder(model.d_num_bands).to(device)
        points, t = sample_stratified(rays)  # (H, W, num_samples, 3)
        points = points.to(device)
        t = t.to(device)
        # (H, W, num_samples, 3 + 3 * 2 * x_num_bands)
        x_encoded = x_encoder(points).to(device)
        # normalize the directions
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True).to(device)
        # (H, W, 3 + 3 * 2 * d_num_bands)
        d_encoded: torch.Tensor = d_encoder(rays_d).to(device)
        # (H, W, num_samples, 3 + 3 * 2 * x_num_bands + 3 + 3 * 2 * d_num_bands)
        d_encoded: torch.Tensor = d_encoded[..., None, :].expand(
            x_encoded.shape[:-1] + (d_encoded.shape[-1],)).to(device)
        input = torch.cat([x_encoded, d_encoded], dim=-1).to(device)

        optimizer.zero_grad()
        output = model(input).to(device)
        rgb, sigma = output[..., :3].to(device), output[..., 3].to(device)

        # TODO: render image
        image = volume_render(rgb, sigma, t, rays_d).to(device)
        target_image = target_image / 255.0

        # remove alpha channel; (4, H, W) to (3, H, W)
        target_image = target_image[:3, :, :]
        # (H, W, 3)
        target_image = target_image.permute(1, 2, 0)

        loss = F.mse_loss(image, target_image).to(device)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(frame), len(dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        # idx += 1


def main():
    device = torch.device("mps")
    model = NeRF(10, 4).to(device)
    train_loader = get_data_loader(
        "data/nerf_synthetic/lego/transforms_train_100x100.json", train=True, shuffle=False, batch_size=None)

    # train the model
    for epoch in range(1):
        train(model, device, train_loader, torch.optim.Adam(
            model.parameters()), epoch)


if __name__ == '__main__':
    main()
