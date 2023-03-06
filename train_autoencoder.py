"""Pointnet Autoencoder training."""

# Third party modules
import numpy as np
import torch
from torch import optim
from torch.utils import data
from tqdm import tqdm

# Local modules
from autoencoder.pointnetAE import PCAutoEncoder
from chamferdist import ChamferDistance
from constants_AE import (
    AE_BEST_PTH,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LATENT_SIZE,
    LEARNING_RATE,
    NUM_POINTS,
    SHAPE,
)

from utils.make_data import Data
from utils.utils import plot_loss

pointcloud_set = Data(SHAPE, NUM_POINTS)
train_dataloader = data.DataLoader(pointcloud_set, batch_size=BATCH_SIZE, shuffle=True)
model = PCAutoEncoder(LATENT_SIZE, NUM_POINTS)
model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
loss_dict = {"chamfer_dist": []}
chamferDist = ChamferDistance()
best_loss = float("inf")

for epoch in range(EPOCHS):
    running_loss = []

    for pointclouds in tqdm(train_dataloader):

        pointclouds = pointclouds.to(DEVICE)
        outputs = model(pointclouds)
        loss = 0.5 * chamferDist(outputs, pointclouds, bidirectional=True)
        loss += 0.5 * chamferDist(pointclouds, outputs, bidirectional=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

    avg_running_loss = np.mean(running_loss)
    loss_dict["chamfer_dist"].append(avg_running_loss)

    if best_loss > avg_running_loss:
        best_loss = avg_running_loss
        torch.save(model.state_dict(), AE_BEST_PTH)

    print(
        "Epoch: "
        + str(epoch + 1)
        + "\ttotal_epochs: "
        + str(EPOCHS)
        + "\tChamfer Distance_loss:"
        + str(round(avg_running_loss, 4))
    )

plot_loss(loss_dict, "autoencoder_loss", "figures/")
