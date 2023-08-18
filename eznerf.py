import os

import torch
import torch.backends.mps
from torch.utils.data import DataLoader

from animate.animate import animate
from config.config import EZNeRFConfig, parse_args
from dataloader.data import NeRFDataset
from model.model import NeRF
from run.run import test_func, train_func


def main(config: EZNeRFConfig):
    coarse_model = NeRF(x_num_bands=10, d_num_bands=4, hidden_dim=256)
    fine_model = NeRF(x_num_bands=10, d_num_bands=4, hidden_dim=256)
    parameters = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    epoch = resume(coarse_model, fine_model, optimizer, config.checkpoint)

    # Setup run functions
    train, validate, test = None, None, None
    if config.train_json:
        print(f"Starting at epoch {epoch}")
        train_dataset = NeRFDataset(config.train_json, config.H, config.W, train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator(device),
        )
        train = train_func(coarse_model, fine_model, train_loader, optimizer)
    if config.val_json:
        val_dataset = NeRFDataset(config.val_json, config.H, config.W, train=False)
        val_loader = DataLoader(
            val_dataset, batch_size=config.H * config.W, shuffle=False
        )
        validate = test_func(coarse_model, fine_model, val_loader, config.output_dir)
    if config.test_json:
        test_dataset = NeRFDataset(config.test_json, config.H, config.W, train=False)
        test_loader = DataLoader(
            test_dataset, batch_size=config.H * config.W, shuffle=False
        )
        test = test_func(coarse_model, fine_model, test_loader, config.output_dir)

    if config.train_json:
        for epoch in range(epoch, epoch + config.epochs):
            train(epoch) if train else None
            torch.save(
                {
                    "epoch": epoch,
                    "coarse_model_state_dict": coarse_model.state_dict(),
                    "fine_model_state_dict": fine_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{config.checkpoints_dir}/checkpoint_{epoch}.pt",
            )
            validate(epoch) if validate else None
            scheduler.step()
        print("Training complete")

    if config.test_json:
        print("Testing")
        test(epoch) if test else None

    if config.animate:
        print("Generating animation")
        animate(coarse_model, device)


def resume(
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
) -> int:
    epoch = 0
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
        fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        epoch += checkpoint["epoch"] + 1
    return epoch


def setup_device(arg_device: str | None) -> torch.device:
    if arg_device is not None:
        device = torch.device(arg_device)
    else:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    print(f"Using device: {device}")
    return device


if __name__ == "__main__":
    args = parse_args()
    device = setup_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    with device:
        main(args)
