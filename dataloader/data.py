import json
import os
from typing import TypedDict

import torch
import torch.utils.data
import torchvision.transforms.functional
from torchvision import io
from tqdm import tqdm

from rays.rays import get_rays


class JSONFrame(TypedDict):
    """A dictionary corresponding to a frame in the NeRF dataset JSON."""

    file_path: str
    transform_matrix: list[list[float]]


class NeRFDataset(torch.utils.data.TensorDataset):
    """A PyTorch Dataset for NeRF."""

    def __init__(self, path: str, H: int, W: int, train: bool):
        """Initializes the dataset.

        Args:
            path (str): The path to the JSON file containing fields:
                camera_angle_x: float
                frames: list of dicts with the following fields
                    file_path: str
                    transform_matrix: list of lists of floats
            train (bool): Whether this is a training or testing loader.
                If training, the dataset will be shuffled for each epoch.
                If testing, the depth and normal maps will be loaded.
        """
        super().__init__()
        self.path = path
        self.train = train
        self.H = H
        self.W = W
        with open(path, "r") as f:
            data = json.load(f)
            self.camera_angle_x: float = data["camera_angle_x"]
            path_frames: list[JSONFrame] = data["frames"]
            self.ray_origins = torch.empty((H * W * len(path_frames), 3))
            self.ray_directions = torch.empty((H * W * len(path_frames), 3))
            self.ray_directions_normalized = torch.empty((H * W * len(path_frames), 3))
            self.images = torch.empty((H * W * len(path_frames), 3))
            print(f"Loading {len(path_frames)} frames from {path}")
            for index, frame in enumerate(tqdm(path_frames)):
                base_path = os.path.join(
                    os.path.dirname(self.path), os.path.relpath(frame["file_path"])
                )
                image = io.read_image(base_path + ".png", io.ImageReadMode.RGB).to(
                    self.ray_origins.device
                )  # [3, H, W]
                image = torchvision.transforms.functional.resize(
                    image, [self.H, self.W], antialias=True
                )
                image = image.permute(1, 2, 0)  # [H, W, 3]
                image = image.float() / 255.0
                pose_matrix = torch.tensor(frame["transform_matrix"])
                ray_origins, ray_directions, directions_normalized = get_rays(
                    image.shape[0], image.shape[1], self.camera_angle_x, pose_matrix
                )
                self.ray_origins[
                    index * H * W : (index + 1) * H * W
                ] = ray_origins.reshape(-1, 3)
                self.ray_directions[
                    index * H * W : (index + 1) * H * W
                ] = ray_directions.reshape(-1, 3)
                self.ray_directions_normalized[
                    index * H * W : (index + 1) * H * W
                ] = directions_normalized.reshape(-1, 3)
                self.images[index * H * W : (index + 1) * H * W] = image.reshape(-1, 3)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.ray_origins.shape[0]

    def __getitem__(self, idx: int):
        """Returns the sample in the dataset at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the following fields:
                image: Tensor of shape (3, 256, 256)
                transform_matrix: Tensor of shape (4, 4)
                depth: Tensor of shape (256, 256)
                normal: Tensor of shape (3, 256, 256)
        """
        return (
            self.ray_origins[idx],
            self.ray_directions[idx],
            self.ray_directions_normalized[idx],
            self.images[idx],
        )
