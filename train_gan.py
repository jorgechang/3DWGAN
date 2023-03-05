"""WGAN training."""

# Third party modules
import numpy as np
import torch
from gan.gan import Discriminator, Generator
from torch.utils import data
from tqdm import tqdm

# Local modules
from constants_WGAN import (
    EPOCHS,
    NUM_POINTS,
    LATENT_SIZE,
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    SHAPE,
    GRADIENT_PENALTY,
    CRITIC_BOOST,
    MU,
    SIGMA
)
from utils.gan_utils import compute_gradient_penalty
from utils.make_data import Data
from utils.utils import noiseFunc, plot_loss, saveModel

training_set_ae = Data(SHAPE, NUM_POINTS)
train_dataloader = data.DataLoader(training_set_ae, batch_size=BATCH_SIZE, shuffle=True)


generator = Generator(NUM_POINTS, LATENT_SIZE)
discriminator = Discriminator(LATENT_SIZE)
generator.to(DEVICE)
discriminator.to(DEVICE)

optimizer_g = torch.optim.AdamW(
    generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
)
optimizer_d = torch.optim.AdamW(
    discriminator.parameters(), lr=LEARNING_RATE+.0001, betas=(0.5, 0.999)
)

loss_dict = {}
total_iters = 0
loss_dict[generator.name] = []
loss_dict[discriminator.name] = []
best_generator_loss = float('inf')

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    running_loss_generator = []
    running_loss_discriminator = []

    for pointclouds in tqdm(train_dataloader):
        total_iters += 1
        optimizer_d.zero_grad()
        pointclouds = pointclouds.to(DEVICE)
        fake_pointclouds = noiseFunc(
            MU, SIGMA, pointclouds.shape[0], DEVICE, LATENT_SIZE
        )

        # Generate fake data.
        x_fake = generator(fake_pointclouds)

        # train critic
        fake_output = discriminator(x_fake.detach())
        real_output = discriminator(pointclouds.detach())
        x_out = torch.cat((real_output, fake_output))
        d_loss = (
            -(real_output.mean() - fake_output.mean())
            + compute_gradient_penalty(discriminator, pointclouds, x_fake, DEVICE)
            * GRADIENT_PENALTY
        )
        d_loss.backward()
        optimizer_d.step()
        running_loss_discriminator.append(d_loss.item())

        # Train Generator
        if total_iters % CRITIC_BOOST == 0:
            optimizer_g.zero_grad()
            fake_pointclouds = noiseFunc(
                MU, SIGMA, pointclouds.shape[0], DEVICE, LATENT_SIZE
            )
            # Generate fake data.
            x_fake = generator(fake_pointclouds)
            fake_output = discriminator(x_fake)
            g_loss = -fake_output.mean()
            g_loss.backward()
            optimizer_g.step()
            running_loss_generator.append(g_loss.item())

    mean_generator_loss = np.mean(running_loss_generator)
    mean_discriminator_loss = np.mean(running_loss_discriminator)
    loss_dict[generator.name].append(mean_generator_loss)   
    loss_dict[discriminator.name].append(mean_discriminator_loss)

    print("Epoch: " + str(epoch + 1)
        + "\ttotal_epochs: " + str(EPOCHS)
        + "\td_loss:" + str(round(mean_discriminator_loss, 4))
        + "\tg_loss:" + str(round(mean_generator_loss, 4))
        )
    
    if best_generator_loss > mean_generator_loss:
        best_generator_loss = mean_generator_loss
        saveModel("weights/generator/"+'gan_best.pth', generator)
        saveModel("weights/discriminator/"+ 'gan_best.pth', discriminator)

    #plotPointCloud(SHAPE, NUM_POINTS, generator)
    
plot_loss(loss_dict, "WGAN", "figures/")
saveModel("weights/generator/"+'gan_last.pth', generator)
saveModel("weights/discriminator/"+ 'gan_last.pth', discriminator)
