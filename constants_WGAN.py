"""WGAN training constants."""

EPOCHS = 300
NUM_POINTS = 1024
LATENT_SIZE = 128
GUIDED_SIZE = 0
AE_LATENT_SIZE = 16
BATCH_SIZE = 128
DEVICE = "cuda"
LEARNING_RATE = 0.00005
SHAPE = "chair"
GRADIENT_PENALTY = 30
CRITIC_BOOST = 5
AE_BEST_PTH = "models/autoencoder/best.pth"
MU = 0.0
SIGMA = 0.2
if GUIDED_SIZE == 0:
    WGAN_NAME = "latent"
else:
    WGAN_NAME = "latent_guided"