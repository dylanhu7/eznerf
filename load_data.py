import numpy as np
import torch
import os
import torch.utils.data
import json
from typing import TypedDict
from torchvision import io


class PathFrame(TypedDict):
    file_path: str
    rotation: float
    transform_matrix: list[list[float]]


class ImageFrame(TypedDict):
    image: torch.Tensor
    transform_matrix: torch.Tensor
    rotation: float


class TestFrame(TypedDict):
    image: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor
    transform_matrix: torch.Tensor
    rotation: float


class NeRFDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for NeRF."""

    def __init__(self, path: str, batch_size: int, train: bool):
        """Initializes the dataset.

        Args:
            path (str): The path to the JSON file containing fields:
                camera_angle_x: float
                frames: list of dicts with the following fields
                    file_path: str
                    rotation: float
                    transform_matrix: list of lists of floats
            batch_size (int): The batch size.
            train (bool): Whether this is a training or testing loader.
                If training, the dataset will be shuffled for each epoch. If testing, the depth and normal maps will be loaded.
        """
        self.path = path
        self.batch_size = batch_size
        self.train = train
        # Read the JSON file and get the camera angle and frames.
        with open(path, 'r') as f:
            data = json.load(f)
            self.camera_angle_x: float = data['camera_angle_x']
            self.frames: list[PathFrame] = data['frames']

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> ImageFrame | TestFrame:
        """Returns the sample in the dataset at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the following fields:
                image: torch.Tensor of shape (3, 256, 256)
                depth: torch.Tensor of shape (256, 256)
                normal: torch.Tensor of shape (3, 256, 256)
                transform_matrix: torch.Tensor of shape (4, 4)
                rotation: float
        """
        # Get the frame.
        frame = self.frames[idx]
        # Read the image.
        image = io.read_image(os.path.join(os.path.dirname(
            self.path), os.path.relpath(frame['file_path'] + '.png')))
        # Get the transform matrix.
        transform_matrix = torch.tensor(frame['transform_matrix'])
        # Get the rotation.
        rotation = frame['rotation']
        # If this is a training loader, return the image and transform matrix.
        if self.train:
            return {
                'image': image,
                'transform_matrix': transform_matrix,
                'rotation': rotation
            }
        # If this is a testing loader, return the image, depth, normal, transform matrix, and rotation.
        else:
            return {
                'image': image,
                'depth': io.read_image(os.path.join(os.path.dirname(
                    self.path), os.path.relpath(frame['file_path'] + '_depth_0001.png'))),
                'normal': io.read_image(os.path.join(os.path.dirname(
                    self.path), os.path.relpath(frame['file_path'] + '_normal_0001.png'))),
                'transform_matrix': transform_matrix,
                'rotation': rotation
            }


def get_data_loader(path: str, batch_size: int, train: bool) -> tuple[torch.utils.data.DataLoader, NeRFDataset]:
    """Returns a PyTorch DataLoader for the NeRF dataset.

    Args:
        path (str): The path to the JSON file containing fields:
            camera_angle_x: float
            frames: list of dicts with the following fields
                file_path: str
                rotation: float
                transform_matrix: list of lists of floats
        batch_size (int): The batch size.
        train (bool): Whether this is a training or testing loader.
            If training, the dataset will be shuffled for each epoch. If testing, the depth and normal maps will be loaded.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """
    dataset = NeRFDataset(path, batch_size, train)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=4)
    return data_loader, dataset
