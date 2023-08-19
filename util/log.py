import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from config.config import EZNeRFConfig
from model.model import NeRF
from run.run import test_func


def log_batch(
    config: EZNeRFConfig,
    coarse_model: NeRF,
    fine_model: NeRF,
    optimizer: torch.optim.Optimizer,
    pbar: tqdm,
    val_loader: DataLoader | None = None,
):
    validate = test_func(config, coarse_model, fine_model)

    def log(
        loss: Tensor,
        psnr: Tensor,
        coarse_loss: Tensor,
        fine_loss: Tensor,
        i: int,
    ):
        pbar.set_description(
            f"[Training | Loss: {loss.item():.4f} | PSNR: {psnr.item():.4f}]"
        )
        if i % config.log_interval == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "psnr": psnr.item(),
                    "coarse_loss": coarse_loss.item(),
                    "fine_loss": fine_loss.item(),
                },
                step=i,
            )
        if i % config.checkpoint_interval == 0:
            checkpoint_path = f"{config.checkpoints_dir}/checkpoint_{i}.pt"
            torch.save(
                {
                    "iter": i,
                    "coarse_model_state_dict": coarse_model.state_dict(),
                    "fine_model_state_dict": fine_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            wandb.save(checkpoint_path)
        if i % config.val_interval == 0 and val_loader is not None:
            pbar.set_description(
                f"[Validating | Loss: {loss.item():.4f} | PSNR: {psnr.item():.4f}]"
            )
            batch = next(iter(val_loader))
            image, val_loss, val_psnr = validate(batch)
            wandb.log(
                {
                    "val_loss": val_loss.item(),
                    "val_psnr": val_psnr.item(),
                    "val_image": wandb.Image(image),
                },
                step=i,
            )

    return log
