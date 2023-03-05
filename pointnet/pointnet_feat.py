"""
MIT License

Copyright (c) 2017 Fei Xia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Third party modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local modules
from pointnet.tnet import TNet3d


class PointNetfeat(nn.Module):
    """Pointnet feature extractor."""

    def __init__(self, latent_size, global_feat=True, feature_transform=False):
        """Init section."""
        super(PointNetfeat, self).__init__()
        self.stn = TNet3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, latent_size, 1)
        self.bn1 = nn.LayerNorm(64)
        self.bn2 = nn.LayerNorm(128)
        self.bn3 = nn.LayerNorm(latent_size)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.latent_size = latent_size

    def forward(self, x):
        """Forward pass."""
        x = x.permute(0, 2, 1)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)

        return x, trans, trans_feat
