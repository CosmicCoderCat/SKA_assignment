import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_dirty_clean(dirty_img: np.ndarray, clean_img: np.ndarray, vmin: float, vmax: float, extra_label: Optional[str] = None, save: bool = True, output_dir: Optional[str] = None) -> None:
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

    filename = f'{output_dir}/dirty_vs_clean.png' if output_dir else 'dirty_vs_clean.png'
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

def plot_amplitude_vs_time(binned_amp: np.ndarray, time_bins: np.ndarray, title: str, filename: str, outlier_mask: Optional[np.array] = None, amplitude_median: Optional[float] = None, amplitude_mad: Optional[float] = None, multiplier: Optional[int] = None, save: bool = True, output_dir: Optional[str] = None) -> None:
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
    filename: str
        Filename to save the plot
    outlier_mask: Optional[np.array], optional
        Mask for outliers, not plotted by default None.
    amplitude_median: Optional[float], optional
        Median of the visibility amplitude, not plotted by default None.
    amplitude_mad: Optional[float], optional
        Median absoulute deviation of the visibility amplitude, not plotted by default None.
    multiplier: Optional[int], optional
        Multiplier for the median absolute deviation, marking flagging region, not plotted by default None.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_bins, binned_amp, marker='o', linestyle='-', zorder=1)

    if amplitude_median is not None:
        ax.plot(time_bins, np.full(len(time_bins), amplitude_median), color='orange', label='Median', zorder=10)

    if amplitude_mad is not None:
        ax.fill_between(time_bins,
                        amplitude_median - multiplier * amplitude_mad,
                        amplitude_median + multiplier * amplitude_mad,
                        color='orange', alpha=0.3, label=f'{multiplier}x Median Abs Deviation', zorder=20)

    if outlier_mask is not None:
        ax.scatter(np.array(time_bins)[outlier_mask], binned_amp[outlier_mask], marker='x', color='red', linestyle='-', label='Amplitude', zorder=30)

    ax.set_xlabel("Time Bins")
    ax.set_ylabel("Average Visibility Amplitude")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    filename = f'{output_dir}/{filename}' if output_dir else filename
    if save:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()