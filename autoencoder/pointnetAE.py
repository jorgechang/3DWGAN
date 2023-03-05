"""Pointnet autoencoder."""

# Third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules
from pointnet.pointnet_feat import PointNetfeat

class Decoder(nn.Module):
    """Point Cloud Autoencoder.

    Attributes:
        num_points (int):
            Number of input points.

        latent_size (int):
            Autoencoder latent size.
    """

    def __init__(self, num_points, latent_size):
        """Init section."""
        super(Decoder, self).__init__()
        self.name = "decoder"
        self.fc1 = nn.Linear(in_features=latent_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=num_points * 3)

        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(1024)

        self.num_points = num_points

    def forward(self, x):
        """Forward pass."""
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = F.gelu(self.ln3(self.fc3(x)))
        x = F.gelu(self.fc4(x))
        x = self.fc5(x)
        return torch.reshape(x, (-1, self.num_points, 3))


class PCAutoEncoder(nn.Module):
    """Point Cloud Autoencoder.

    Attributes:
        latent_size (int):
            Autoencoder latent size.
            Default: 128

        num_points (int):
            Number of input points.
            Default: 1024
    """

    def __init__(self, latent_size=128, num_points=1024):
        """Init section."""
        super(PCAutoEncoder, self).__init__()
        self.encoder = PointNetfeat(latent_size)
        self.decoder = Decoder(num_points, latent_size)
        self.num_points = num_points

    def forward(self, x):
        """Forward pass."""
        x, _, _ = self.encoder(x)
        reconstructed_points = self.decoder(x)
        return torch.reshape(reconstructed_points, (-1, self.num_points, 3))
