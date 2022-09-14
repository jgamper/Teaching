from typing import Dict
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle



def generate_circles(num_samples: int = 20000, scale: float = 0.005) -> pd.DataFrame:

    t = np.linspace(0, 2, num_samples)
    x = np.sin(np.pi * t) + np.random.normal(0, scale, num_samples)
    y = np.cos(np.pi * t) + np.random.normal(0, scale, num_samples)
    label = np.ones(num_samples)

    return pd.DataFrame({"label": label, "x": x, "y": y})


def generate_data(num_samples: int = 20000, scale: float = 0.005) -> pd.DataFrame:

    # Generate first cirlce
    d1 = generate_circles(num_samples, 0.05)

    # Generate and shift the second circle
    d2 = generate_circles(num_samples, 0.05)
    d2.x, d2.y, d2.label = 0.5 * d2.x, 0.5 * d2.y, 0 * d2.label

    return pd.concat([d1, d2])


def plot_scatter(data: pd.DataFrame):
    plt.rcParams["axes.labelcolor"] = "white"
    plt.rcParams["xtick.labelcolor"] = "white"
    plt.rcParams["ytick.labelcolor"] = "white"
    DARKBLUE = "#1d1330"
    ORANGE = "#ffa72b"
    LIGHTBLUE = "#e1f7fa"

    colors = np.where(data.label == 1, "darkgreen", "darkorange")

    fig, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.set_axis_off()
    ax.scatter(data.x, data.y, c=colors)
    return fig, ax

def sample_data(data, N):
    """
    Shuffles data frame and samples N rows
    """
    df_local = data.copy()
    df_local = shuffle(df_local)

    out = df_local.sample(N)[['x', 'y', 'label']].to_numpy()
    inp = out[:, :2]
    label = out[:,2]
    inp = torch.Tensor(inp)
    label = torch.Tensor(out[:,2])

    if torch.cuda.is_available():
        return inp.cuda(), label.cuda()
    else:
        return inp, label