import os
import argparse
import glob

import torch
import torch.backends.mps

from train.train import train_func
from model.model import NeRF
from dataloader.data import get_data_loader


def main():
    device = setup_device()
    model = NeRF(10, 4).to(device)
    train_loader = get_data_loader(
        "data/nerf_synthetic/lego/transforms_train_40x40.json", train=True, shuffle=True, batch_size=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # # Find latest checkpoint
    epoch: int = 0
    # checkpoints = glob.glob("checkpoints/checkpoint_*.pt")
    # if len(checkpoints) > 0:
    #     checkpoints.sort()
    #     checkpoint = torch.load(checkpoints[-1])
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     epoch = checkpoint['epoch']
    #     print(f"Loaded checkpoint {checkpoints[-1]}")

    # Stage train function
    train = train_func(model, device, train_loader, optimizer)

    # Train the model
    for epoch in range(epoch + 1, 1000):
        train(epoch)
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoints/checkpoint_{epoch}.pt')


def setup_device() -> torch.device:
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('output'):
        os.makedirs('output')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    return device


if __name__ == '__main__':
    main()
