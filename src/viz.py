import numpy as np
import matplotlib.pyplot as plt

def plot_grid(grid, title="Grid"):
    fig, ax = plt.subplots()
    ax.imshow(grid)
    ax.set_title(title)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    return fig

def plot_loss(losses, title="Training Loss"):
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    return fig

def growth_map(built0, built1):
    return ((built1 == 1) & (built0 == 0)).astype(int)
