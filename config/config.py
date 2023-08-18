from typing import Optional
import configargparse


class EZNeRFConfig(configargparse.Namespace):
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


def parse_args() -> EZNeRFConfig:
    parser = configargparse.ArgumentParser()
    parser.add_argument("-c", "--config", is_config_file=True)
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--test_json", type=str, default=None)
    parser.add_argument("--H", type=int, default=400)
    parser.add_argument("--W", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None)

    args: EZNeRFConfig = parser.parse_args(namespace=EZNeRFConfig())

    return args
