import os

import torch
import torch.backends.mps
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from animate.animate import animate
from config.config import EZNeRFConfig, get_config
from dataloader.data import NeRFDataset
from model.model import NeRF
from run.run import train_batch
from util.infinite_iterable import InfiniteIterable
from util.log import log_batch


def main(config: EZNeRFConfig):
    coarse_model = NeRF(x_num_bands=10, d_num_bands=4, hidden_dim=256)
    fine_model = NeRF(x_num_bands=10, d_num_bands=4, hidden_dim=256)
    parameters = list(coarse_model.parameters()) + list(fine_model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=torch.exp(torch.log(torch.tensor(0.1)) / 250000).item()
    )
    initial_iter = resume(coarse_model, fine_model, optimizer, config.checkpoint)

    if config.train_json:
        if initial_iter == 0:
            print("Training from scratch")
        else:
            print(f"Resuming training from iteration {initial_iter}")
        train_dataset = NeRFDataset(config.train_json, config.H, config.W, train=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator(device),
        )
        val_loader = None
        if config.val_json:
            val_dataset = NeRFDataset(config.val_json, config.H, config.W, train=False)
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.H * config.W,
                shuffle=False,
            )
        iterable = InfiniteIterable(train_loader, config.iters)
        pbar = tqdm(iterable, initial=initial_iter, total=config.iters + initial_iter)
        train = train_batch(coarse_model, fine_model, optimizer, scheduler)
        log = log_batch(config, coarse_model, fine_model, optimizer, pbar, val_loader)
        for i, batch in enumerate(pbar):
            i = i + initial_iter
            loss, psnr, coarse_loss, fine_loss = train(batch)
            log(loss, psnr, coarse_loss, fine_loss, i)

    if config.animate:
        print("Generating animation")
        animate(coarse_model, device)


def resume(
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
) -> int:
    iteration = 0
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
        fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from iteration {checkpoint['iter']}")
        iteration += checkpoint["iter"] + 1
    return iteration


def setup_device(config: EZNeRFConfig) -> torch.device:
    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    print(f"Using device: {device}")
    config.device = device.__str__()
    return device


if __name__ == "__main__":
    config = get_config()
    device = setup_device(config)
    wandb.init(project="eznerf", config=config, anonymous="allow")  # type: ignore
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    with device:
        main(config)
