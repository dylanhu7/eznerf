from dataclasses import dataclass


@dataclass
class EZNeRFConfig:
    train_json: str
    val_json: str
    H: int
    W: int
    checkpoint: str | None
    checkpoints_dir: str
    iters: int
    batch_size: int
    lr: float
    coarse_x_num_bands: int
    coarse_d_num_bands: int
    coarse_hidden_dim: int
    fine_x_num_bands: int
    fine_d_num_bands: int
    fine_hidden_dim: int
    device: str | None
    seed: int | None
    log_interval: int
    val_interval: int
    checkpoint_interval: int
