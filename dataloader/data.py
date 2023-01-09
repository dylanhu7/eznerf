import torch
import os
import torch.utils.data
import json
from typing import TypedDict, Optional
from torchvision import io


class PathFrame(TypedDict):
    """A dictionary corresponding to a frame in the NeRF dataset JSON."""
    file_path: str
    transform_matrix: list[list[float]]


class Frame(TypedDict):
    """A dictionary corresponding to a frame in the NeRF dataset for training."""
    image: torch.Tensor
    transform_matrix: torch.Tensor
    depth: Optional[torch.Tensor]
    normal: Optional[torch.Tensor]


class TestFrame(TypedDict):
    """A dictionary corresponding to a frame in the NeRF dataset for testing."""
    image: torch.Tensor
    transform_matrix: torch.Tensor
    depth: torch.Tensor
    normal: torch.Tensor


class NeRFDataset(torch.utils.data.Dataset[Frame]):
    """A PyTorch Dataset for NeRF."""

    def __init__(self, path: str, train: bool):
        """Initializes the dataset.

        Args:
            path (str): The path to the JSON file containing fields:
                camera_angle_x: float
                frames: list of dicts with the following fields
                    file_path: str
                    transform_matrix: list of lists of floats
            train (bool): Whether this is a training or testing loader.
                If training, the dataset will be shuffled for each epoch. If testing, the depth and normal maps will be loaded.
        """
        super().__init__()
        self.path = path
        self.train = train
        # Read the JSON file and get the camera angle and frames.
        with open(path, 'r') as f:
            data = json.load(f)
            self.camera_angle_x: float = data['camera_angle_x']
            self.frames: list[PathFrame] = data['frames']
            self.image_size: tuple[int, int] = io.read_image(os.path.join(os.path.dirname(
                self.path), os.path.relpath(self.frames[0]['file_path'] + '.png'))).shape[1:]
            self.image_height: int = self.image_size[0]
            self.image_width: int = self.image_size[1]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
        """Returns the sample in the dataset at the given index.

        Args:
            idx (int): The index of the sample.

        Returns:
            dict: A dictionary containing the following fields:
                image: torch.Tensor of shape (3, 256, 256)
                transform_matrix: torch.Tensor of shape (4, 4)
                depth: torch.Tensor of shape (256, 256)
                normal: torch.Tensor of shape (3, 256, 256)
        """
        # Get the frame.
        frame = self.frames[idx]
        # Read the image.
        image = io.read_image(os.path.join(os.path.dirname(
            self.path), os.path.relpath(frame['file_path'] + '.png')))
        # Get the transform matrix.
        transform_matrix = torch.tensor(frame['transform_matrix'])
        # Get the rotation.
        # If this is a training loader, return the image and transform matrix.
        if self.train:
            return {
                'image': image,
                'transform_matrix': transform_matrix,
                'depth': None,
                'normal': None
            }
        # If this is a testing loader, return the image, depth, normal, transform matrix, and rotation.
        else:
            return {
                'image': image,
                'transform_matrix': transform_matrix,
                'depth': io.read_image(os.path.join(os.path.dirname(
                    self.path), os.path.relpath(frame['file_path'] + '_depth_0001.png'))),
                'normal': io.read_image(os.path.join(os.path.dirname(
                    self.path), os.path.relpath(frame['file_path'] + '_normal_0001.png'))),
            }


def get_data_loader(path: str, batch_size: Optional[int], train=True, shuffle=True, num_workers=0) -> torch.utils.data.DataLoader[Frame]:
    """Returns a PyTorch DataLoader for the NeRF dataset.

    Args:
        path (str): The path to the JSON file containing fields:
            camera_angle_x: float
            frames: list of dicts with the following fields
                file_path: str
                transform_matrix: list of lists of floats
        batch_size (int, optional): The batch size.
            If None, automatic batching will be disabled.
        train (bool): Whether this is a training or testing loader.
            If training, the dataset will be shuffled for each epoch. If testing, the depth and normal maps will be loaded.
            Defaults to True.
        shuffle (bool): Whether to shuffle the dataset for each epoch.
            Defaults to True.
        num_workers (int): The number of workers to use for loading the data.
            Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: The data loader.
    """
    dataset = NeRFDataset(path, train)
    data_loader = torch.utils.data.DataLoader[Frame](
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
