import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.backends.mps
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from load_data import get_data_loader
from encoder import Encoder


class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        self.encoder = Encoder(num_bands=10, max_freq=10)

    def forward(self, x):
        pass


def main():
    model = NeRF()
    train_loader, train_dataset = get_data_loader(
        "data/nerf_synthetic/lego/transforms_train.json", 100, False)
    test_loader, test_dataset = get_data_loader(
        "data/nerf_synthetic/lego/transforms_test.json", 100, False)


if __name__ == '__main__':
    main()
