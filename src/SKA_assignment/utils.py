import numpy as np


def get_binned_visibility_amplitude(
    vis: np.ndarray,
    masks: np.ndarray,
    first_freq_step: int = 0,
    n_freq_steps: int = 32,
    freq_step: int = 1,
) -> np.ndarray:
    """Returns the binned visibility amplitude for given visibility data set
    and for given mask array.

    Parameters
    ----------
    vis : numpy.ndarray
        visibility data
    masks : np.ndarray
        Mask array for the visibility dta.
    first_freq_step : int, optional
        Used to select a subset of channels to bin.
        Frequency channel to start bin from, by default 0.
    n_freq_steps : int, optional
        Used to select a subset of channels to bin.
        How many channels to include, by default all 32.
    freq_step : int, optional
        Used to select a subset of channels to bin.
        How many channels to include in a bin, by default 1.

    Returns
    -------
    np.ndarray
        Binned visibility amplitude array.
    """
    # Get the amplitude of the visibilities and group them by frequency instead
    amplitude_all_vis = np.abs(vis).T

    # Extract the requested frequency slices
    amplitude_vis = amplitude_all_vis[
        first_freq_step : first_freq_step + n_freq_steps * freq_step, :
    ]

    # Compute binned amplitudes for all frequencies and time bins in one go
    binned_amplitude = np.array(
        [
            [
                np.mean(
                    amplitude_vis[freq_index : freq_index + freq_step][mask],
                    axis=0,
                )
                if np.any(mask)
                else np.nan
                for mask in masks
            ]
            for freq_index in range(n_freq_steps)
        ]
    )

    return binned_amplitude


def get_combined_masks(
    time_all: np.ndarray,
    unique_times: np.ndarray,
    dt: float,
    autocorr_filter: np.ndarray,
    first_t_frame: int,
    nb_t_steps: int,
    t_step: int,
) -> np.ndarray:
    """Given a subset of time frames and an autocorrrelation filter, returns an
    array of masks for each time frame.

    Parameters
    ----------
    time_all : np.ndarray
        Time array for the visibilities.
    unique_times : np.ndarray
        The set of unique time values in time_all.
    dt : float
        The interval between successive time values.
    autocorr_filter : np.ndarray
        Autocorrelation filter.
    first_t_frame : int, optional
        Used to select a subset of time frames to analyse.
        Time frame to start analysis from, by default 0.
    nb_t_steps : int, optional
        Used to select a subset of time frames to analyse.
        How many time frames to include, by default all 120.
    t_step : int, optional
        Used to select a subset of time frames to analyse.
        How many time frames to include in an averaged step, by default 1.

    Returns
    -------
    np.ndarray
        Array of masks for each time frame.
    """
    # Create a mask to extract the requested time slices for each frequency
    combined_masks = []
    for i, start_frame in enumerate(
        range(first_t_frame, first_t_frame + nb_t_steps * t_step, t_step)
    ):
        end_frame = start_frame + t_step
        t_start = unique_times[start_frame]
        try:
            t_end = unique_times[end_frame]
        except IndexError:
            t_end = unique_times[-1] + dt

        # Include the autocorrelation filter in the mask
        combined_masks.append(
            [(~autocorr_filter) & (time_all >= t_start) & (time_all < t_end)]
        )

    return combined_masks
