import os
import torch.utils.data
import json
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from typing import TypedDict, Optional
from torchvision import io
from rays.rays import get_rays


class PathFrame(TypedDict):
    """A dictionary corresponding to a frame in the NeRF dataset JSON."""
    file_path: str
    transform_matrix: list[list[float]]


class Frame(TypedDict):
    """A dictionary corresponding to a frame in the NeRF dataset."""
    image: Tensor
    transform_matrix: Tensor
    depth: Optional[Tensor]
    normal: Optional[Tensor]
    rays: Tensor


class NeRFDataset(Dataset[Frame]):
    """A PyTorch Dataset for NeRF."""

    def __init__(self, path: str, train: bool, device: torch.device):
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
        self.frames: list[Frame] = []
        with open(path, 'r') as f:
            data = json.load(f)
            self.camera_angle_x: float = data['camera_angle_x']
            path_frames: list[PathFrame] = data['frames']
            for frame in path_frames:
                base_path = os.path.join(os.path.dirname(self.path), os.path.relpath(frame['file_path']))
                image = io.read_image(base_path + '.png', io.ImageReadMode.RGB).to(device) # [3, H, W]
                image = image.permute(1, 2, 0) # [H, W, 3]
                image = image.float() / 255.0
                transform_matrix = Tensor(frame['transform_matrix']).to(device)
                depth = io.read_image(base_path + '_depth_0001.png').to(device) if not train else None
                normal = io.read_image(base_path + '_normal_0001.png').to(device) if not train else None
                rays = get_rays(image.shape[0], image.shape[1], self.camera_angle_x, transform_matrix)
                self.frames.append({
                    'image': image,
                    'transform_matrix': transform_matrix,
                    'depth': depth,
                    'normal': normal,
                    'rays': rays
                })
        self.image_height = self.frames[0]['image'].shape[0]
        self.image_width = self.frames[0]['image'].shape[1]
        print(f'Loaded {len(self.frames)} frames of size {self.image_width}x{self.image_height} from {self.path}')
            

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.frames)

    def __getitem__(self, idx: int) -> Frame:
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
        return self.frames[idx]
    