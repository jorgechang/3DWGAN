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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TNet3d(nn.Module):
    """Predict an affine transformation matrix by a mini-network."""

    def __init__(self):
        """Init section."""
        super(TNet3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        """Forward pass."""
        batchsize = x.size()[0]
        # [batch_size, 3, n]
        x = F.relu(self.bn1(self.conv1(x)))
        # [batch_size, 64, n]
        x = F.relu(self.bn2(self.conv2(x)))
        # [batch_size, 128, n]
        x = F.relu(self.bn3(self.conv3(x)))
        # [batch_size, 1024, n]
        x = torch.max(x, 2, keepdim=True)[0]
        # [batch_size, 1024, 1]
        x = x.view(-1, 1024)
        # [batch_size, 1024]
        x = F.relu(self.bn4(self.fc1(x)))
        # [batch_size, 512]
        x = F.relu(self.bn5(self.fc2(x)))
        # [batch_size, 256]
        x = self.fc3(x)
        # [batch_size, 9]
        iden = (
            Variable(
                torch.from_numpy(
                    np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)
                )
            )
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        # [batch_size, 9]
        x = x.view(-1, 3, 3)
        # [batch_size, 3, 3]
        return x
