import argparse
import glob
import os
from typing import Optional

import torch
import torch.backends.mps
from tqdm import tqdm

from animate.animate import animate
from dataloader.data import get_train_loader, get_test_loader
from model.model import NeRF
from run.train import run_func


def main():
    args = parse_args()
    device = setup_device(args.device)
    initialize_directories(args.output_dir, args.checkpoints_dir)

    model = NeRF(10, 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    epoch: int = 0

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
    elif args.resume:
        checkpoints = glob.glob("checkpoints/checkpoint_*.pt")
        if len(checkpoints) > 0:
            checkpoints.sort()
            checkpoint = torch.load(checkpoints[-1])
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Loaded checkpoint")

    if args.train:
        epoch += 1
        print(f"Starting at epoch {epoch}")
        print(f"Training for {args.epochs} epochs")
        train_loader = get_train_loader(args.train_json)
        train = run_func(model, device, train_loader,
                         optimizer, args.output_dir, train=True)
        test_loader = get_test_loader(args.test_json)
        test = run_func(model, device, test_loader,
                        optimizer, args.output_dir, train=False)
        for epoch in tqdm(range(epoch, epoch + args.epochs)):
            train(epoch)
            if epoch % 20 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f'checkpoints/checkpoint_{epoch}.pt')
                test(epoch)
        print("Training complete")

    if args.test:
        test_loader = get_test_loader(args.test_json)
        test = run_func(model, device, test_loader,
                        optimizer, args.output_dir, train=False)
        print("Testing")
        test(epoch)

    if args.animate:
        print("Generating animation")
        animate(model, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--train_json', type=str, default=None)
    parser.add_argument('--test_json', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--checkpoints_dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    if args.train and args.train_json is None:
        raise ValueError("Must specify train json file to enable training")
    if args.test and args.test_json is None:
        raise ValueError("Must specify test json file to enable testing")

    if args.resume and args.checkpoint is not None:
        raise ValueError("Cannot resume and load checkpoint at the same time")

    if args.epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0")

    return args


def setup_device(arg_device: Optional[str]) -> torch.device:
    if arg_device is not None:
        device = torch.device(arg_device)
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    return device


def initialize_directories(output_dir: str, checkpoints_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)


if __name__ == '__main__':
    main()
