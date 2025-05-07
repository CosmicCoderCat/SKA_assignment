import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_dirty_clean(dirty_img: np.ndarray, clean_img: np.ndarray, vmin: float, vmax: float, extra_label: Optional[str] = None) -> None:
    """
    Plot the dirty and cleaned images side by side.

    Parameters
    ----------
    dirty_img : np.ndarray
        The dirty image array to be plotted.
    clean_img : np.ndarray
        The cleaned image array to be plotted.
    vmin : float
        Minimum intensity value for the color scale.
    vmax : float
        Maximum intensity value for the color scale.
    extra_label : Optional[str], optional
        Additional label to include in the plot titles, by default None.
    """
    fig, axs = plt.subplots(1, 2, figsize=(18, 10))

    im0 = axs[0].imshow(dirty_img, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Dirty Image\n{extra_label}")
    axs[0].set_xlabel("Pixel X")
    axs[0].set_ylabel("Pixel Y")
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04).set_label("Intensity")

    im1 = axs[1].imshow(clean_img, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Cleaned Image\n{extra_label}")
    axs[1].set_xlabel("Pixel X")
    axs[1].set_ylabel("Pixel Y")
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04).set_label("Intensity")

    plt.tight_layout()
    plt.show()

def plot_amplitude_vs_time(binned_amp: np.ndarray, time_bins: np.ndarray, title: str) -> None:
    """
    Plot average visibility amplitude vs time bins.

    Parameters
    ----------
    binned_amp : np.ndarray
        Array of average visibility amplitudes for each time bin.
    time_bins : np.ndarray
        Array of time bin centers.
    title : str
        Title of the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time_bins, binned_amp, marker='o', linestyle='-')
    plt.xlabel("Time Bins")
    plt.ylabel("Average Visibility Amplitude")
    plt.title(title)
    plt.show()