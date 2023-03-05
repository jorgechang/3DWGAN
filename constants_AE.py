"""Autoencoder training constants."""

EPOCHS = 500
NUM_POINTS = 1024
LATENT_SIZE = 16
BATCH_SIZE = 128
DEVICE = "cuda"
LEARNING_RATE = 0.0001
SHAPE = "chair"
LOSS = "Chamfer Distance"
AE_BEST_PTH = "weights/autoencoder/best.pth"
VISUALIZE = False