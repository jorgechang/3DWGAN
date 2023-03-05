"""GAN components."""

# Third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules
from pointnet.pointnet_feat import PointNetfeat


class Generator(nn.Module):
    """Point Cloud Generator.

    Attributes:
        num_points (int):
            Number of output points.

        latent_size (int):
            random noise size.

        guided_latent_size (int):
            latent space guiding size.
            Default: 0
    """

    def __init__(self, num_points, latent_size, guided_latent_size=0):
        """Init section."""
        super(Generator, self).__init__()
        self.name = "Generator"
        self.fc1 = nn.Linear(
            in_features=latent_size + guided_latent_size, out_features=256
        )
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=1024)
        self.fc4 = nn.Linear(in_features=1024, out_features=1024)
        self.fc5 = nn.Linear(in_features=1024, out_features=num_points * 3)

        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(1024)

        self.num_points = num_points
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        """Forward pass."""
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        x = F.gelu(self.ln3(self.fc3(x)))
        x = self.activation(self.fc4(x))
        x = self.fc5(x)
        return torch.reshape(x, (-1, self.num_points, 3))


class Discriminator(nn.Module):
    """Point Cloud Discriminator.

    Attributes:
        num_features (int):
            Number of global features.
    """

    def __init__(self, num_features):
        """Init section."""
        super(Discriminator, self).__init__()
        self.name = "Discriminator"
        self.pointnet_features = PointNetfeat(num_features)
        self.dense = nn.Linear(num_features, 1)

    def forward(self, x):
        """Forward pass."""
        x = self.pointnet_features(x)[0]
        return self.dense(x)
