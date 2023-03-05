"""Latent guided WGAN training."""

# Third party modules
import numpy as np
import torch
from autoencoder.pointnetAE import PCAutoEncoder
from chamferdist import ChamferDistance
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
    AE_LATENT_SIZE,
    GUIDED_SIZE,
    MU,
    SIGMA,
    WGAN_NAME
)
from utils.gan_utils import compute_gradient_penalty
from utils.make_data import Data
from utils.utils import noiseFunc, plot_loss, plotPointCloud, saveModel

training_set_ae = Data(SHAPE, NUM_POINTS)
train_dataloader = data.DataLoader(training_set_ae, batch_size=BATCH_SIZE, shuffle= True)

generator = Generator(NUM_POINTS,LATENT_SIZE,GUIDED_SIZE)
discriminator = Discriminator(LATENT_SIZE)
generator.to(DEVICE)
discriminator.to(DEVICE)

autoencoder = PCAutoEncoder(AE_LATENT_SIZE,NUM_POINTS)
autoencoder.load_state_dict(torch.load("weights/autoencoder/best.pth"))
autoencoder.to(DEVICE)
autoencoder.eval()
encoder = autoencoder.encoder

optimizer_g = torch.optim.AdamW(generator.parameters(),
                                   lr=LEARNING_RATE,
                                   betas=(0.5, 0.999))
optimizer_d = torch.optim.AdamW(discriminator.parameters(),
                                   lr=LEARNING_RATE,
                                   betas=(0.5, 0.999))

chamferDist = ChamferDistance()
loss_dict = {}
total_iters = 0
loss_dict[generator.name] = []
loss_dict[discriminator.name] = []
best_generator_loss = float('inf')

for epoch in range(EPOCHS): 
    running_loss_generator = []
    running_loss_discriminator = []

    for pointclouds in tqdm(train_dataloader):
        total_iters += 1
        optimizer_d.zero_grad()

        pointclouds = pointclouds.to(DEVICE)
        fake_pointclouds = noiseFunc(MU, SIGMA, pointclouds.shape[0], DEVICE, LATENT_SIZE)
        
        # Adds latent representation to noise if training guided GAN.
        if GUIDED_SIZE != 0:
            guided_latent = encoder(pointclouds)[0]
            fake_pointclouds = torch.cat([fake_pointclouds, guided_latent], dim=1)
        
        # Generate fake data.
        x_fake = generator(fake_pointclouds)
        #train critic
        fake_output = discriminator(x_fake.detach())
        real_output = discriminator(pointclouds.detach())
        x_out = torch.cat((real_output, fake_output))
        d_loss = -(real_output.mean() - fake_output.mean()) + compute_gradient_penalty(discriminator, pointclouds, x_fake, DEVICE) * GRADIENT_PENALTY + (x_out ** 2).mean() * 0.0001
        d_loss.backward()
        optimizer_d.step()
        running_loss_discriminator.append(d_loss.item())

        # Train Generator
        if total_iters % CRITIC_BOOST == 0:
            optimizer_g.zero_grad()
            fake_pointclouds = noiseFunc(MU, SIGMA, pointclouds.shape[0], DEVICE, LATENT_SIZE)
            if GUIDED_SIZE != 0:
                guided_latent = encoder(pointclouds)[0]
                fake_pointclouds = torch.cat((fake_pointclouds,guided_latent), dim=1)

            x_fake = generator(fake_pointclouds)
            fake_output = discriminator(x_fake)
            cdloss = 0.5 * chamferDist(x_fake, pointclouds, bidirectional=True)
            cdloss += 0.5 * chamferDist(pointclouds, x_fake, bidirectional=True)
            g_loss = - fake_output.mean() + cdloss
            g_loss.backward()
            optimizer_g.step()
            running_loss_generator.append(g_loss.item())

    mean_generator_loss = np.mean(running_loss_generator)
    mean_discriminator_loss = np.mean(running_loss_discriminator)

    loss_dict[generator.name].append(mean_generator_loss)
    loss_dict[discriminator.name].append(mean_discriminator_loss)

    #lr_scheduler.step()

    print("Epoch: " + str(epoch + 1)
            + "\ttotal_epochs: " + str(EPOCHS)
            + "\td_loss:" + str(round(mean_discriminator_loss, 4))
            + "\tg_loss:" + str(round(mean_generator_loss, 4))
            )

    if best_generator_loss > mean_generator_loss:
        best_generator_loss = mean_generator_loss
        saveModel(f'weights/generator/{WGAN_NAME}_best.pth', generator)
        saveModel(f'weights/discriminator/{WGAN_NAME}_best.pth', discriminator)

    #plotPointCloud(SHAPE, NUM_POINTS, generator, encoder)

plot_loss(loss_dict, WGAN_NAME+"_loss", "figures/")
saveModel(f'weights/generator/{WGAN_NAME}_last.pth', generator)
saveModel(f'weights/discriminator/{WGAN_NAME}_last.pth', discriminator)
