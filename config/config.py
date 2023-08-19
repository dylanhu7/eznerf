import configargparse


class EZNeRFConfig(configargparse.Namespace):
    train_json: str | None
    val_json: str | None
    test_json: str | None
    H: int
    W: int
    checkpoint: str | None
    checkpoints_dir: str
    iters: int
    batch_size: int
    animate: bool
    device: str
    seed: int
    log_interval: int
    val_interval: int
    checkpoint_interval: int


def get_config() -> EZNeRFConfig:
    parser = configargparse.ArgumentParser(
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--H", type=int, default=400)
    parser.add_argument("--W", type=int, default=400)
    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--val_interval", type=int, default=500)
    parser.add_argument("--checkpoint_interval", type=int, default=1000)

    config: EZNeRFConfig = parser.parse_args(namespace=EZNeRFConfig())

    return config
