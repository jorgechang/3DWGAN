"""Utils from https://github.com/square-1111/3D-Point-Cloud-Modeling."""
import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import torch
from constants_WGAN import GUIDED_SIZE
from IPython import display
from plotly.subplots import make_subplots
from utils.make_data import Data


# Function to generate noise vector
def noiseFunc(mu, sigma, batch_size, device, num_point):
    """
    Function to generate noise from Gaussian Distribution

    Args:
    mu, sigma : Mean and Standard Deviation for Normal Distribution
    device    : GPU or CPU device
    """
    return torch.normal(mu, sigma, (batch_size, num_point)).float().to(device)


def plot_loss(losses, model_name, path="./"):
    """
    A function to visualize loss to see changes in loss

    Args:
    losses (dict)   : A list of dictionary
    model_name(str) : Name of model along with criterion
    path(str)       : Path where to save loss visualization
    """
    loss_path = os.path.join(path, model_name)
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10, 8))
    plt.title(model_name)
    colors = {
        "Generator": "blue",
        "Discriminator": "red",
        "chamfer_dist": "blue",
    }
    for key, value in losses.items():
        plt.plot(value, label=key, color=colors[key])
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("{}_loss.png".format(loss_path))
    # plt.show()


def plotPointCloud(object, num_points, model=None, encoder=None):
    """Retrieve 3 random point clouds and plot interactive graphs."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.3,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )

    objlst = Data(object, num_points)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mu = 0
    sigma = 0.2
    with torch.no_grad():
        for i in range(2):
            point_cloud = noiseFunc(mu, sigma, 1, device, 128)
            if GUIDED_SIZE != 0:
                guided_example = objlst[np.random.randint(len(objlst))]
                guided_example = encoder(guided_example[None, :].to(device))[0]
                point_cloud = torch.cat((point_cloud, guided_example), dim=1)
            point_cloud = torch.squeeze(point_cloud, 0)
            point_cloud = model(point_cloud[None, :].to(device))
            point_cloud = torch.squeeze(point_cloud, 0)
            np_point_cloud = point_cloud.detach().cpu().numpy()

            fig.add_trace(
                go.Scatter3d(
                    x=np_point_cloud[:, 0],
                    y=np_point_cloud[:, 1],
                    z=np_point_cloud[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color=np_point_cloud[:, 2],
                        colorscale="Viridis",
                        opacity=1,
                    ),
                ),
                row=1,
                col=i + 1,
            )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20), showlegend=False, width=800, height=400
        )

        pio.write_image(fig, "point_clouds.png")

def saveModel(path, model):
    """
    Save the model and its weight to specified path

    Attributes:
    path : location where to save model
    model : weighted model
    epoch : save epoch every 10 epochs
    """
    torch.save(model.state_dict(), path)
