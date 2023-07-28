import argparse
import glob
import os
from typing import Optional

import torch
import torch.backends.mps
from natsort import natsorted
from torch.utils.data import DataLoader

from animate.animate import animate
from dataloader.data import NeRFDataset
from model.model import NeRF
from run.run import train_func, test_func


class EZNeRFArgs(argparse.Namespace):
    train_json: Optional[str]
    val_json: Optional[str]
    test_json: Optional[str]
    H: int
    W: int
    checkpoint: Optional[str]
    resume: bool
    checkpoints_dir: str
    output_dir: str
    epochs: int
    batch_size: int
    animate: bool
    device: str


def main(args: EZNeRFArgs):
    coarse_model = NeRF(x_num_bands=10, d_num_bands=4, hidden_dim=256)
    fine_model = NeRF(x_num_bands=10, d_num_bands=4, hidden_dim=256)
    parameters = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
    epoch = resume(coarse_model, fine_model, optimizer, args.checkpoint)

    # Setup run functions
    train, validate, test = None, None, None
    if args.train_json:
        print(f"Starting at epoch {epoch}")
        train_dataset = NeRFDataset(args.train_json, args.H, args.W, train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator(device),
        )
        train = train_func(coarse_model, fine_model, train_loader, optimizer)
    if args.val_json:
        val_dataset = NeRFDataset(args.val_json, args.H, args.W, train=False)
        val_loader = DataLoader(val_dataset, batch_size=args.H * args.W, shuffle=False)
        validate = test_func(coarse_model, fine_model, val_loader, args.output_dir)
    if args.test_json:
        test_dataset = NeRFDataset(args.test_json, args.H, args.W, train=False)
        test_loader = DataLoader(
            test_dataset, batch_size=args.H * args.W, shuffle=False
        )
        test = test_func(coarse_model, fine_model, test_loader, args.output_dir)

    if args.train_json:
        for epoch in range(epoch, epoch + args.epochs):
            train(epoch) if train else None
            torch.save(
                {
                    "epoch": epoch,
                    "coarse_model_state_dict": coarse_model.state_dict(),
                    "fine_model_state_dict": fine_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{args.checkpoints_dir}/checkpoint_{epoch}.pt",
            )
            validate(epoch) if validate else None
            scheduler.step()
        print("Training complete")

    if args.test_json:
        print("Testing")
        test(epoch) if test else None

    if args.animate:
        print("Generating animation")
        animate(coarse_model, device)


def resume(
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Optional[str],
) -> int:
    epoch = 0
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
    elif args.resume:
        checkpoints = natsorted(glob.glob(os.path.join(args.checkpoints_dir, "*.pt")))
        if len(checkpoints) > 0:
            checkpoint = torch.load(checkpoints[-1])
    if checkpoint is not None:
        coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
        fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        epoch += checkpoint["epoch"] + 1
    return epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--H", type=int, default=400)
    parser.add_argument("--W", type=int, default=400)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args(namespace=EZNeRFArgs())

    if args.train_json is None and args.test_json is None and not args.animate:
        raise ValueError("Must provide at least one of: train_json, test_json, animate")

    if args.train_json is not None and args.epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0 if training")

    if args.H <= 0 or args.W <= 0:
        raise ValueError("Height and width must be greater than 0")

    if args.animate and args.checkpoint is None and not args.resume:
        raise ValueError("Must provide a checkpoint or --resume flag to animate")

    if args.resume and args.checkpoint is not None:
        raise ValueError("Cannot resume and load checkpoint at the same time")

    if args.epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0")

    return args


def setup_device(arg_device: Optional[str]) -> torch.device:
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
