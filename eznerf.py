import logging
import os
import warnings

import PIL.Image as Image
import torch
import torch.backends.mps
import torch.nn.functional as F
import typer
from rich.logging import RichHandler
from torch.utils.data import DataLoader
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

import wandb
from config.config import EZNeRFConfig
from dataloader.data import NeRFDataset
from model.model import NeRF
from render.render import render_rays
from util.looping_dataloader import LoopingDataLoader

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Configure logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)
logger = logging.getLogger("eznerf")

app = typer.Typer()


@app.command()
def train(
    train_json: str = "data/nerf_synthetic/lego/transforms_train.json",
    val_json: str = "data/nerf_synthetic/lego/transforms_val.json",
    H: int = 400,
    W: int = 400,
    checkpoint: str | None = None,
    checkpoints_dir: str = "checkpoints",
    iters: int = 200000,
    batch_size: int = 1024,
    lr: float = 5e-4,
    coarse_x_num_bands: int = 10,
    coarse_d_num_bands: int = 4,
    coarse_hidden_dim: int = 256,
    fine_x_num_bands: int = 10,
    fine_d_num_bands: int = 4,
    fine_hidden_dim: int = 256,
    device: str | None = None,
    seed: int | None = None,
    log_interval: int = 100,
    val_interval: int = 500,
    checkpoint_interval: int = 1000,
):
    if seed is not None:
        torch.manual_seed(seed)

    # Create a config object from the arguments.
    config = EZNeRFConfig(**locals())

    wandb.init(
        project="eznerf",
        config=config.__dict__,
        anonymous="allow",
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    device_obj = setup_device(device)
    with device_obj:
        coarse_model = NeRF(
            x_num_bands=coarse_x_num_bands,
            d_num_bands=coarse_d_num_bands,
            hidden_dim=coarse_hidden_dim,
        )
        fine_model = NeRF(
            x_num_bands=fine_x_num_bands,
            d_num_bands=fine_d_num_bands,
            hidden_dim=fine_hidden_dim,
        )
        parameters = list(coarse_model.parameters()) + list(fine_model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1000, gamma=0.5
        )

        initial_iter = resume(
            coarse_model, fine_model, optimizer, scheduler, checkpoint
        )

        train_dataset = NeRFDataset(train_json, H, W, train=True)
        train_loader = LoopingDataLoader(
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=torch.Generator(device_obj),
            ),
            max_iter=iters - initial_iter,
        )
        val_dataset = NeRFDataset(val_json, H, W, train=False)
        val_loader = LoopingDataLoader(
            DataLoader(
                val_dataset,
                batch_size=H * W,
                shuffle=False,
                generator=torch.Generator(device_obj),
            )
        )

        logger.info("Training...")
        for i, batch in enumerate(
            tqdm(train_loader, initial=initial_iter, total=iters),
            start=initial_iter,
        ):
            # Training step
            loss, psnr, coarse_loss, fine_loss = train_batch(
                coarse_model, fine_model, optimizer, scheduler, batch
            )

            # Logging
            if i % log_interval == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "psnr": psnr.item(),
                        "coarse_loss": coarse_loss.item(),
                        "fine_loss": fine_loss.item(),
                    },
                    step=i,
                )
            if i % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoints_dir}/checkpoint_{i}.pt"
                torch.save(
                    {
                        "iter": i,
                        "coarse_model_state_dict": coarse_model.state_dict(),
                        "fine_model_state_dict": fine_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    checkpoint_path,
                )
                wandb.save(checkpoint_path)
            if i % val_interval == 0:
                val_batch = next(iter(val_loader))
                image, val_loss, val_psnr = test_batch(
                    config, coarse_model, fine_model, val_batch
                )
                wandb.log(
                    {
                        "val_loss": val_loss.item(),
                        "val_psnr": val_psnr.item(),
                        "val_image": wandb.Image(image),
                    },
                    step=i,
                )


def train_batch(
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ray_origins, ray_directions, ray_directions_normalized, target_image = batch

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

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss, psnr, coarse_loss, fine_loss


@torch.no_grad()
def test_batch(
    config: EZNeRFConfig,
    coarse_model: NeRF,
    fine_model: NeRF,
    batch: tuple[torch.Tensor, ...],
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    ray_origins, ray_directions, ray_directions_normalized, target_image = batch

    chunk_size = config.batch_size * 2
    chunk_count = ray_origins.shape[0] // chunk_size + 1
    chunk_rays_o = torch.chunk(ray_origins, chunk_count)
    chunk_rays_d = torch.chunk(ray_directions, chunk_count)
    chunk_rays_d_norm = torch.chunk(ray_directions_normalized, chunk_count)
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
    psnr = -10 * torch.log10(loss)
    image = image.reshape(config.H, config.W, 3)
    image = (torch.clip(image, 0, 1) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)

    return image, loss, psnr


def resume(
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_path: str | None,
) -> int:
    iteration = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        coarse_model.load_state_dict(checkpoint["coarse_model_state_dict"])
        fine_model.load_state_dict(checkpoint["fine_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Loaded checkpoint from iteration {checkpoint['iter']}")
        iteration += checkpoint["iter"] + 1
    if iteration == 0:
        logger.info("Training from scratch")
    else:
        logger.info(f"Resuming training from iteration {iteration}")
    return iteration


def setup_device(device_str: str | None) -> torch.device:
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        )
    logger.info(f"Using device: [bold cyan]{device}[/]")
    return device


if __name__ == "__main__":
    app()
